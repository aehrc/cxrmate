import torch
import torch.nn.functional as F

from task.mimic_cxr.model.report_gen.any.prompt import GTPrompt
from task.mimic_cxr.model.report_gen.any.single import SingleCXR
from task.mimic_cxr.model.report_gen.any.variable import VariableCXR


class GTPromptMatchingTokenTypeIDs(GTPrompt):
    """
    Prompt the decoder with the findings and impression section of the previous study.

    Use the same token type identifiers as the current study.
    """
    def __init__(self, **kwargs):
        VariableCXR.__init__(self, **kwargs)  # For two token type identifiers.

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """

        # Tokenize report:
        tokenized = self.tokenize_report_teacher_forcing(batch['findings'], batch['impression'])

        # Tokenize prompt:
        prompt = self.tokenize_prompt(batch['previous_findings'], batch['previous_impression'])

        # Joint the token identifiers:
        decoder_input_ids = torch.cat(
            [prompt['input_ids'], tokenized['decoder_input_ids']], dim=1,
        )
        decoder_attention_mask = torch.cat(
            [prompt['attention_mask'], tokenized['decoder_attention_mask']], dim=1,
        )

        # Get the position identifiers:
        decoder_position_ids = torch.nn.functional.relu(
            torch.cumsum(decoder_attention_mask, dim=1, dtype=torch.int64) - 1
        )

        # Get token type identifiers:
        token_type_ids = self.token_ids_to_token_type_ids(
            decoder_input_ids, 
            [
                self.tokenizer.additional_special_tokens_ids[
                    self.tokenizer.additional_special_tokens.index('[PMT-SEP]')
                ],
                self.tokenizer.bos_token_id,
                self.tokenizer.sep_token_id,
            ],
            [0, 1, 0, 1]
        )

        # Inference
        y_hat = self(            
            images=batch['images'], 
            dicom_study_ids=batch['dicom_study_ids'], 
            decoder_input_ids=decoder_input_ids, 
            decoder_attention_mask=decoder_attention_mask, 
            decoder_token_type_ids=token_type_ids,
            decoder_position_ids=decoder_position_ids,
        )

        # Add padding to account for prompt:
        label_ids = F.pad(
            tokenized['label_ids'],
            (y_hat.shape[1] - tokenized['label_ids'].shape[1], 0, 0, 0),
            'constant',
            self.tokenizer.pad_token_id,
        )

        # Loss:
        loss = F.cross_entropy(
            y_hat.permute([0, 2, 1]), label_ids, ignore_index=self.tokenizer.pad_token_id,
        )

        # Logging:
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        # Update and log scores for each validation metric:
        return loss

    @staticmethod
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
            token_type_ids = SingleCXR.token_ids_to_token_type_ids(input_ids, special_token_ids, [0, 1, 0, 1])
        else:
            token_type_ids = SingleCXR.token_ids_to_token_type_ids_past(input_ids, special_token_ids, [0, 1, 0, 1])
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
