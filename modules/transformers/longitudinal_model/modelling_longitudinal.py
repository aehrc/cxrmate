import os
import warnings
from typing import Any, Optional, Tuple, Union

import torch
import transformers
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, PreTrainedTokenizerFast,
                          VisionEncoderDecoderModel)
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.models.vision_encoder_decoder.configuration_vision_encoder_decoder import \
    VisionEncoderDecoderConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


class CvtWithProjectionHeadConfig(transformers.CvtConfig):
    def __init__(self, projection_size: int = None, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.projection_size = projection_size


class ModelOutputWithProjectionEmbedding(transformers.modeling_outputs.ModelOutput):
    last_hidden_state: torch.FloatTensor
    attention_mask: torch.FloatTensor


class CvtProjectionHead(torch.nn.Module):

    def __init__(self, config) -> None:
        super().__init__()

        # https://github.com/huggingface/transformers/blob/68287689f2f0d8b7063c400230b3766987abf18d/src/transformers/models/cvt/modeling_cvt.py#L657
        self.layer_norm = torch.nn.LayerNorm(config.embed_dim[-1], eps=config.layer_norm_eps)

        # No bias as following layer normalisation with bias:
        self.projection = torch.nn.Linear(config.embed_dim[-1], config.projection_size, bias=False)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norm(x)
        x = self.projection(x)
        return x


class VariableCvtWithProjectionHead(transformers.CvtPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.cvt = transformers.CvtModel(config, add_pooling_layer=False)
        self.projection_head = CvtProjectionHead(config)

        # Initialize weights and apply final processing:
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[torch.Tensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, ModelOutputWithProjectionEmbedding]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Flatten the batch and study_id dimensions:
        outputs = self.cvt(
            pixel_values.view(-1, *pixel_values.shape[2:]),
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Flatten h x w:
        last_hidden_state = torch.flatten(outputs.last_hidden_state, 2)

        # Project the features for each spatial position to the decoder's hidden size:
        projection = self.projection_head(torch.permute(last_hidden_state, [0, 2, 1]))

        # Concatenate the features for each chest X-ray:
        projection = projection.view(pixel_values.shape[0], -1, projection.shape[-1])

        # Derive the attention mask from the pixel values:
        attention_mask = (pixel_values[:, :, 0, 0, 0] != 0.0).repeat_interleave(last_hidden_state.shape[-1], dim=1)

        if not return_dict:
            return projection

        return ModelOutputWithProjectionEmbedding(
            last_hidden_state=projection, attention_mask=attention_mask,
        )
    

class LongitudinalPromptVariableCXREncoderDecoderModel(VisionEncoderDecoderModel):

    config_class = VisionEncoderDecoderConfig
    base_model_prefix = "vision_encoder_decoder"
    main_input_name = "pixel_values"
    supports_gradient_checkpointing = True

    def __init__(        
        self,
        config: Optional[PretrainedConfig] = None,
        encoder: Optional[PreTrainedModel] = None,
        decoder: Optional[PreTrainedModel] = None,
        encoder_decoder_ckpt_name: Optional[str] = None,
    ):

        if decoder:
            assert decoder.config.add_cross_attention, '"add_cross_attention" must be True for the given decoder'
            assert decoder.config.is_decoder, '"is_decoder" must be True for the given decoder'

        if config is None and (encoder is None or decoder is None):
            raise ValueError("Either a configuration or an encoder and a decoder has to be provided.")
        if config is None:
            config = VisionEncoderDecoderConfig.from_encoder_decoder_configs(encoder.config, decoder.config)
        else:
            if not isinstance(config, self.config_class):
                raise ValueError(f"Config: {config} has to be of type {self.config_class}")

        config.tie_word_embeddings = False

        # initialize with config
        PreTrainedModel.__init__(self, config)

        # Encoder:
        if encoder is None:
            encoder = VariableCvtWithProjectionHead(config=config.encoder)

        # Decoder:
        if decoder is None:
            decoder = transformers.BertLMHeadModel(config=config.decoder)

        self.encoder = encoder
        self.decoder = decoder

        if self.encoder.config.to_dict() != self.config.encoder.to_dict():
            logger.warning(
                f"Config of the encoder: {self.encoder.__class__} is overwritten by shared encoder config:"
                f" {self.config.encoder}"
            )
        if self.decoder.config.to_dict() != self.config.decoder.to_dict():
            logger.warning(
                f"Config of the decoder: {self.decoder.__class__} is overwritten by shared decoder config:"
                f" {self.config.decoder}"
            )
            
        self.encoder.config = self.config.encoder
        self.decoder.config = self.config.decoder

        # Load variable checkpoint:
        if encoder_decoder_ckpt_name:
            encoder_decoder = AutoModel.from_pretrained(encoder_decoder_ckpt_name, trust_remote_code=True)
            self.load_state_dict(encoder_decoder.state_dict())
        else:
            warnings.warn('The encoder-to-decoder model was not warm-started before applying low-rank approximation.')

        # Freeze the encoder:
        for p in self.encoder.parameters():
            p.requires_grad = False
            
        # Freeze the decoder and add LoRA:
        peft_config = LoraConfig(
            inference_mode=False, 
            r=8, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules='bert.encoder.layer.[0-9]+.attention.self.(query|key)',
        )
        self.decoder = get_peft_model(self.decoder, peft_config)
        self.decoder.print_trainable_parameters()

    def forward(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.BoolTensor] = None,
        encoder_outputs: Optional[Tuple[torch.FloatTensor]] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple[torch.FloatTensor], Seq2SeqLMOutput]:

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_") :]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }

        if encoder_outputs is None:
            if pixel_values is None:
                raise ValueError("You have to specify pixel_values")

            encoder_outputs = self.encoder(
                pixel_values,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )  # CvT does not support output_attentions.
        elif isinstance(encoder_outputs, tuple):
            encoder_outputs = BaseModelOutput(*encoder_outputs)

        encoder_hidden_states = encoder_outputs[0]
        
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_outputs.attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Loss:
        loss = None
        if labels is not None:
            logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            # encoder_hidden_states=encoder_outputs.hidden_states,
            # encoder_attentions=encoder_outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        special_token_ids,
        mask_token_id,
        past_key_values=None,
        attention_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
    ):
        """
        Modification of: 
            https://github.com/huggingface/transformers/blob/main/src/transformers/models/encoder_decoder/modeling_encoder_decoder.py#L660
        """

        decoder_inputs = self.decoder.prepare_inputs_for_generation(input_ids, past_key_values=past_key_values)
        decoder_attention_mask = (input_ids != mask_token_id).int()
        decoder_position_ids = torch.nn.functional.relu(
            torch.cumsum(decoder_attention_mask, dim=1, dtype=torch.int64) - 1
        )

        if not past_key_values:
            token_type_ids = self.token_ids_to_token_type_ids(input_ids, special_token_ids, [0, 1, 0, 1])
        else:
            token_type_ids = self.token_ids_to_token_type_ids_past(input_ids, special_token_ids, [0, 1, 0, 1])
            decoder_position_ids = decoder_position_ids[:, -1:]

        input_dict = {
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'decoder_input_ids': decoder_inputs['input_ids'],
            'decoder_token_type_ids': token_type_ids,
            'decoder_position_ids': decoder_position_ids,
            'encoder_outputs': encoder_outputs,
            'past_key_values': decoder_inputs['past_key_values'],
            'use_cache': use_cache,
        }
        return input_dict
    
    def token_ids_to_token_type_ids(self, token_ids, special_token_ids, token_type_id_sections=None):
        """
        Extract token type identifiers from the token identifiers.

        Argument/s:
            token_ids - token identifiers.
            special_token_ids - special token identifiers that indicate the separation between sections.
            token_type_id_section - token type identifier for each section.

        Returns:
            token_type_ids - token type identifiers.
        """

        token_type_id_sections = token_type_id_sections if token_type_id_sections is not None else list(range(len(special_token_ids) + 1))

        mbatch_size, seq_len = token_ids.shape
        token_type_ids = torch.full_like(token_ids, token_type_id_sections[0], dtype=torch.long, device=token_ids.device)

        for i, j in enumerate(special_token_ids):
            # Find first occurrence of special tokens that indicate the boundary between sections:
            cols = (token_ids == j).int().argmax(dim=1)
            rows = torch.arange(mbatch_size, device=token_ids.device)

            # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer.create_token_type_ids_from_sequences.example
            cols += 1

            # Ensure that the column index is not out of bounds. If 0, then token_id not present.
            # This is safe as index 0 is always a special token (now equal to 1 due to +1):
            rows = rows[torch.logical_and(cols != 1, cols < seq_len)]
            cols = cols[torch.logical_and(cols != 1, cols < seq_len)]

            # Indices to that correspond to the second sequence:
            if rows.nelement() != 0:
                ids = torch.stack([
                    torch.stack([x, z]) for (x, y) in zip(rows, cols) for z in torch.arange(
                        y, seq_len, device=token_ids.device,
                    )
                ])

                token_type_ids[ids[:, 0], ids[:, 1]] = token_type_id_sections[i + 1]

        return token_type_ids

    def token_ids_to_token_type_ids_past(self, token_ids, special_token_ids, token_type_id_sections=None):
        """
        Extract token type identifiers from the token identifiers if past != None.

        Argument/s:
            token_ids - token identifiers.
            special_token_ids - special token identifiers that indicate the separation between sections.

        Returns:
            token_type_ids - token type identifiers.
        """

        token_type_id_sections = token_type_id_sections if token_type_id_sections is not None else list(range(len(special_token_ids) + 1))
        token_type_ids = torch.full([token_ids.shape[0], 1], token_type_id_sections[0], dtype=torch.long, device=token_ids.device)

        # https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizer.create_token_type_ids_from_sequences.example
        token_ids = token_ids[:, :-1]

        for i, j in enumerate(special_token_ids):

            # Find first occurrence of special token, which indicates the boundary between sections:
            exists = torch.any(token_ids == j, dim=1, keepdim=True)
            token_type_ids[exists] = token_type_id_sections[i + 1]

        return token_type_ids
    
    def tokenize_report_teacher_forcing(self, findings: str, impression: str, tokenizer: PreTrainedTokenizerFast, max_len: int):
        """
        Tokenize the reports and creates the inputs and targets for teacher forcing.

        Argument/s:
            findings - findings section.
            impression - impression section.
            return_token_type_ids - return the token type identifiers.
            tokenizer - Hugging Face tokenizer.
            max_len - maximum number of tokens.

        Returns:
            decoder_input_ids - the token identifiers for the input of the decoder.
            decoder_attention_mask - the attention mask for the decoder_input_ids.
            label_ids - the label token identifiers for the decoder.
        """

        # Prepare the sections for the tokenizer by placing special tokens between each section:
        report = [f'{tokenizer.bos_token}{i}{tokenizer.sep_token}{j}{tokenizer.eos_token}' for i, j in
                  zip(findings, impression)]

        # Tokenize the report:
        tokenized = tokenizer(
            report,
            padding='longest',
            truncation=True,
            max_length=max_len + 1,  # +1 to account for the bias between input and target.
            return_tensors='pt',
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(self.device)

        # Modify for language modelling:
        batch_dict = {

            # Labels for the decoder (shifted right by one for autoregression):
            'label_ids': tokenized['input_ids'][:, 1:].detach().clone(),

            # Remove last token identifier to match the sequence length of the labels:
            'decoder_input_ids': tokenized['input_ids'][:, :-1],

            # Attention mask for the decoder_input_ids (remove first token so that the eos_token_id is not considered):
            'decoder_attention_mask': tokenized['attention_mask'][:, 1:],
        }

        return batch_dict

    def split_and_decode_sections(self, token_ids, special_token_ids, tokenizer: PreTrainedTokenizerFast):
        """
        Split the token identifiers into sections, then convert the token identifiers into strings.

        Argument/s:
            token_ids - token identifiers.
            special_token_ids - special token identifiers that indicate the end of each section.
            tokenizer - Hugging Face tokenizer.

        Returns:
            token_type_ids - token type identifiers.
        """

        _, seq_len = token_ids.shape

        # The number of sections is the same as the number of special_token_ids:
        num_sections = len(special_token_ids)

        sections = {k: [] for k in range(num_sections)}

        for i in token_ids:
            prev_col = 0
            for j, k in enumerate(special_token_ids):

                # The maximum sequence length was exceeded, thus no more tokens:
                if prev_col >= seq_len:
                    sections[j].append('')
                    continue

                # Find first occurrence of special tokens that indicate the boundary between sections:
                col = (i == k).int().argmax().item()

                # If equal to 0, token was not found, set the column to the sequence length (as the decoder exceeded
                # the maximum sequence length):
                if col == 0:
                    col = seq_len

                # Extract section token identifiers:
                section_token_ids = i[prev_col:col]
                prev_col = col
                section_string = tokenizer.decode(section_token_ids, skip_special_tokens=True)

                sections[j].append(section_string)

        return tuple(sections.values())

    def tokenize_prompt(
        self, 
        previous_findings: str, 
        previous_impression: str, 
        tokenizer: PreTrainedTokenizerFast, 
        max_len: int,
        add_bos_token_id: bool = False,
    ):
        """
        Tokenize the sections of the previous report to be used as a prompt.

        Argument/s:
            previous_findings - previous findings section.
            previous_impression - previous impression section.
            tokenizer - Hugging Face tokenizer.
            max_len - maximum number of tokens.
            add_bos_token_id - whether to add the BOS token identifier to the prompt.

        Returns:
            input_ids - the input identifiers for the previous impression.
            attention_mask - the attention mask for the previous impression
        """

        # Use [NPF]/[NPI] special token if no previous findings/impression:
        previous_findings = ['[NPF]' if not i else i for i in previous_findings]
        previous_impression = ['[NPI]' if not i else i for i in previous_impression]

        # Prepare the sections for the tokenizer by placing special tokens:
        previous_sections = [
            f'[PMT]{i}[PMT-SEP]{j}{tokenizer.bos_token}' if add_bos_token_id else f'[PMT]{i}[PMT-SEP]{j}' \
                for i, j in zip(previous_findings, previous_impression)
        ]

        # Tokenize:
        previous_sections = tokenizer(
            previous_sections,
            padding='longest',
            truncation=True,
            max_length=max_len,
            return_tensors='pt',
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(self.device)

        # Ensure BOS token identifier is at the end of the input_ids:
        if previous_sections.input_ids.shape[1] == max_len:
            previous_sections.input_ids[:, -1] = torch.where(
                previous_sections.attention_mask[:, -1] == 1,
                tokenizer.bos_token_id,
                previous_sections.input_ids[:, -1],
            ) 

        assert previous_sections.input_ids.shape[1] <= max_len

        return {'input_ids': previous_sections.input_ids, 'attention_mask': previous_sections.attention_mask}
