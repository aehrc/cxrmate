import os
import types

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from peft import LoraConfig, TaskType, get_peft_config, get_peft_model
from torch.utils.data import DataLoader
from torchvision import transforms

from task.mimic_cxr.datasets.prompt import PreviousReportSubset
from task.mimic_cxr.model.report_gen.any.prompt import GTPrompt
from task.mimic_cxr.model.report_gen.any.single import SingleCXR
from task.mimic_cxr.model.report_gen.any.variable import VariableCXR


class GTPromptLoRA(GTPrompt):
    """
    Prompt the decoder with the findings and impression section of the previous study.

    Use the same token type identifiers as the current study.
    """
    def __init__(self, variable_ckpt_path, lora_rank=8, **kwargs):
        self.variable_ckpt_path = variable_ckpt_path
        self.lora_rank = lora_rank
        VariableCXR.__init__(self, **kwargs)  # For two token type identifiers.


    def init_modules(self):
        """
        Initialise torch.nn.Modules.
        """

        # Decoder tokenizer:
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(self.tokenizer_dir, local_files_only=True)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Print the special tokens:
        print('Description, Special token, Index')
        for k, v in self.tokenizer.special_tokens_map.items():
            if k != 'additional_special_tokens':
                print(f'{k}, {v}, {getattr(self.tokenizer, k + "_id")}')
            else:
                for i, j in zip(self.tokenizer.additional_special_tokens, self.tokenizer.additional_special_tokens_ids):
                    print(f'additional_special_token, {i}, {j}')

        # Decoder; Hugging Face Transformers BERT is used as it accepts token_type_ids as input:
        config = transformers.BertConfig(
            vocab_size=len(self.tokenizer),
            num_hidden_layers=6,
            type_vocab_size=self.type_vocab_size,
        )
        config.add_cross_attention = True
        config.is_decoder = True

        # Do not want this as an attribute of self as it will be an attribute of self.encoder_decoder
        decoder = transformers.BertLMHeadModel(config=config)

        # Use Hugging Face Transformers EncoderDecoderModel to generate conditionally:
        self.encoder_decoder = transformers.EncoderDecoderModel(encoder=decoder.bert, decoder=decoder)
        self.encoder_decoder.decoder.config.hidden_size = self.encoder_decoder.config.decoder.hidden_size

        # Remove encoder from encoder-to-decoder model and replace it with a dummy object
        del self.encoder_decoder.encoder

        # We don't actually want to use the encoder of the EncoderDecoderModel:
        class DummyEncoder:
            main_input_name = 'dummy'

            class DummyConfig:
                pass

            config = DummyConfig()

            def __init__(self, hidden_size):
                self.config.hidden_size = hidden_size

        # Replace the encoder with the dummy encoder:
        self.encoder_decoder.encoder = DummyEncoder(hidden_size=self.encoder_decoder.decoder.config.hidden_size)

        # Overwrite prepare_inputs_for_generation so that token_type_ids are included in input during generation:
        self.encoder_decoder.prepare_inputs_for_generation = types.MethodType(
            self.prepare_inputs_for_generation, self.encoder_decoder,
        )

        # Add this for prepare_inputs_for_generation:
        self.encoder_decoder.eos_token_id = self.tokenizer.eos_token_id
        self.encoder_decoder.sep_token_id = self.tokenizer.sep_token_id

        # Encoder:
        ckpt_name = 'microsoft/cvt-21-384-22k'
        self.encoder = transformers.CvtModel.from_pretrained(
            os.path.join(self.ckpt_zoo_dir, ckpt_name),
            local_files_only=True,
        )

        # https://github.com/huggingface/transformers/blob/b487096b02307cd6e0f132b676cdcc7255fe8e74/src/transformers/models/vit/modeling_vit.py#L505
        self.last_hidden_state_layer_norm = torch.nn.LayerNorm(
            self.encoder.config.embed_dim[-1], eps=self.encoder.config.layer_norm_eps,
        )

        # This is to get the pre-processing parameters for the checkpoint, this is not actually used for pre-processing:
        self.encoder_feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            os.path.join(self.ckpt_zoo_dir, ckpt_name),
            local_files_only=True,
        )

        # Image transformations:
        self.train_transforms = transforms.Compose(
            [
                transforms.Resize(size=self.encoder_feature_extractor.size['shortest_edge']),
                transforms.RandomCrop(
                    size=[
                        self.encoder_feature_extractor.size['shortest_edge'],
                        self.encoder_feature_extractor.size['shortest_edge'],
                    ],
                    pad_if_needed=True,
                ),
                transforms.RandomRotation(degrees=5),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.encoder_feature_extractor.image_mean,
                    std=self.encoder_feature_extractor.image_std,
                ),
            ]
        )
        self.test_transforms = transforms.Compose(
            [
                transforms.Resize(size=self.encoder_feature_extractor.size['shortest_edge']),
                transforms.CenterCrop(size=[
                    self.encoder_feature_extractor.size['shortest_edge'],
                    self.encoder_feature_extractor.size['shortest_edge'],
                ]
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=self.encoder_feature_extractor.image_mean,
                    std=self.encoder_feature_extractor.image_std,
                ),
            ]
        )

        # Encoder projection; no bias as following layer normalisation with bias:
        self.encoder_projection = torch.nn.Linear(
            self.encoder.config.embed_dim[-1], self.encoder_decoder.config.decoder.hidden_size, bias=False,
        )

        # Load variable checkpoint:
        self.load_state_dict(torch.load(self.variable_ckpt_path)['state_dict'])

        # Freeze the encoder:
        for p in self.encoder.parameters():
            p.requires_grad = False
        for p in self.encoder_projection.parameters():
            p.requires_grad = False
        for p in self.last_hidden_state_layer_norm.parameters():
            p.requires_grad = False
            
        # Freeze the decoder and add LoRA:
        peft_config = LoraConfig(
            inference_mode=False, 
            r=self.lora_rank, 
            lora_alpha=32, 
            lora_dropout=0.1, 
            target_modules='bert.encoder.layer.[0-9]+.attention.self.(query|key)',
        )
        self.encoder_decoder.decoder = get_peft_model(self.encoder_decoder.decoder, peft_config)
        self.encoder_decoder.decoder.print_trainable_parameters()

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


class GeneratedPrompt(GTPromptLoRA):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        assert self.mbatch_size == 1

    def setup(self, stage=None):
        """
        https://pytorch-lightning.readthedocs.io/en/stable/common/lightning_module.html#setup
        """

        # Load the merged MIMIC-CXR .csv file:
        df = pd.read_csv(self.merged_csv_path)

        # Dataframe that provides the history for each study:
        history = df.copy()

        # Drop studies that don't have a findings or impression section:
        df = df.dropna(subset=['findings', 'impression'], how='any')

        # Drop studies that have more than the maximum number of DICOMs per study:
        df = df[df.study_id.map(df.study_id.value_counts()) <= self.max_images_per_study]

        if stage == 'fit' or stage is None:
            raise ValueError(f'Only testing can be performed with {self.__class__.__name__}.')

        if stage == 'validate' or stage is None:
            self.val_set = PreviousReportSubset(
                df=df.loc[df['split'] == 'validate'],
                history=history.loc[history['split'] == 'validate'].copy(),
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
                use_generated=True,
                mbatch_size=self.mbatch_size,
            )
            print(f'No. of validation examples: {self.val_set.__len__()}.')
            print(
                f'No. of validation dicom_ids & study_ids: {self.val_set.df.dicom_id.nunique()}',
                f'& {self.val_set.df.study_id.nunique()}.',
            )

        if stage == 'test' or stage is None:
            self.test_set = PreviousReportSubset(
                df=df.loc[df['split'] == 'test'],
                history=history.loc[history['split'] == 'test'].copy(),
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
                use_generated=True,
                mbatch_size=self.mbatch_size,
            )
            print('No. of test examples: {}.'.format(self.test_set.__len__()))
            print(
                f'No. of test dicom_ids & study_ids: {self.test_set.df.dicom_id.nunique()}',
                f'& {self.test_set.df.study_id.nunique()}.',
            )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.mbatch_size,
            num_workers=0,  # Only one worker as generated previous impression will be missed.
            shuffle=False,
            collate_fn=self.collate_fn,
        )  # Cannot use prefetch_factor as generated previous impression will be missed.

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.mbatch_size,
            num_workers=0,  # Only one worker as generated previous impression will be missed.
            shuffle=False,
            collate_fn=self.collate_fn,
        )  # Cannot use prefetch_factor as generated previous impression will be missed.

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Argument/s:
            batch - mini-batch from the validation set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # Tokenize prompt:
        prompt = self.tokenize_prompt(
            batch['previous_findings'], batch['previous_impression'], add_bos_token_id=True,
        )

        # Greedy search:
        output_ids = self.generate(
            num_beams=1, 
            dicom_study_ids=batch['dicom_study_ids'], 
            prompt_ids=prompt['input_ids'],
            images=batch['images'],
        )

        # Findings and impression sections (exclude previous impression section):
        _, findings, impression = self.split_and_decode_sections(
            output_ids,
            [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
        )

        # Add the generated sections the dataframe of the test set:
        for i, j in enumerate(batch['study_ids']):
            self.val_set.history.loc[self.val_set.history.study_id == j, 'generated_findings'] = findings[i]
            self.val_set.history.loc[self.val_set.history.study_id == j, 'generated_impression'] = impression[i]

        # Log reports:
        self.val_report_logger.update(findings, impression, study_ids=batch['study_ids'])

        # Evaluate:
        for i in self.val_metrics:
            if 'findings' in i:
                getattr(self, i).update(
                    findings, [[j] for j in batch['findings']], study_ids=batch['study_ids'],
                )
            elif 'impression' in i:
                getattr(self, i).update(
                    impression, [[j] for j in batch['impression']],  study_ids=batch['study_ids'],
                )
            elif 'report' in i:
                getattr(self, i).update(
                    [f'{i} {j}' for i, j in zip(findings, impression)],
                    [[f'{i} {j}'] for i, j in zip(batch['findings'], batch['impression'])],
                    study_ids=batch['study_ids'],
                )
            else:
                raise ValueError(f'{i} must contain findings, impression, or report')

    def test_step(self, batch, batch_idx):
        """
        Test step.

        Argument/s:
            batch - mini-batch from the test set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # Tokenize prompt:
        prompt = self.tokenize_prompt(
            batch['previous_findings'], batch['previous_impression'], add_bos_token_id=True,
        )

        # Beam search:
        output_ids = self.generate(
            num_beams=self.num_test_beams, 
            dicom_study_ids=batch['dicom_study_ids'], 
            prompt_ids=prompt['input_ids'],
            images=batch['images'],
        )

        # Log report token identifier:
        self.test_report_ids_logger.update(output_ids, study_ids=batch['study_ids'])

        # Findings and impression sections (exclude previous impression section):
        _, findings, impression = self.split_and_decode_sections(
            output_ids,
            [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
        )

        # Add the generated sections the dataframe of the test set:
        for i, j in enumerate(batch['study_ids']):
            self.test_set.history.loc[self.test_set.history.study_id == j, 'generated_findings'] = findings[i]
            self.test_set.history.loc[self.test_set.history.study_id == j, 'generated_impression'] = impression[i]

        # Log reports:
        self.test_report_logger.update(findings, impression, study_ids=batch['study_ids'])

        # Evaluate:
        for i in self.test_metrics:
            if 'findings' in i:
                getattr(self, i).update(
                    findings, [[j] for j in batch['findings']], study_ids=batch['study_ids'],
                )
            elif 'impression' in i:
                getattr(self, i).update(
                    impression, [[j] for j in batch['impression']], study_ids=batch['study_ids'],
                )
            elif 'report' in i:
                getattr(self, i).update(
                    [f'{i} {j}' for i, j in zip(findings, impression)],
                    [[f'{i} {j}'] for i, j in zip(batch['findings'], batch['impression'])],
                    study_ids=batch['study_ids'],
                )
            else:
                raise ValueError(f'{i} must contain findings, impression, or report')
