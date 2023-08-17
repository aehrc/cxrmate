import os
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torchvision import transforms
from transformers.modeling_outputs import BaseModelOutput

from data.prompt import PreviousReportSubset
from modules.lightning_modules.multi import MultiCXR
from modules.transformers.longitudinal_model.modelling_longitudinal import (
    CvtWithProjectionHeadConfig,
    LongitudinalPromptMultiCXREncoderDecoderModel)


class GTPrompt(MultiCXR):
    """
    Prompt the decoder with the findings and impression section of the previous study.
    """
    def __init__(self, multi_ckpt_name, lora_rank=8, **kwargs):
        self.multi_ckpt_name = multi_ckpt_name
        self.lora_rank = lora_rank
        super().__init__(**kwargs)

    def init_modules(self):
        """
        Initialise torch.nn.Modules.
        """

        encoder_decoder_ckpt_name = 'aehrc/cxrmate-tf'

        # Decoder tokenizer:
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(encoder_decoder_ckpt_name, cache_dir=self.ckpt_zoo_dir)
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        # Print the special tokens:
        print('Description, Special token, Index')
        for k, v in self.tokenizer.special_tokens_map.items():
            if k != 'additional_special_tokens':
                print(f'{k}, {v}, {getattr(self.tokenizer, k + "_id")}')
            else:
                for i, j in zip(self.tokenizer.additional_special_tokens, self.tokenizer.additional_special_tokens_ids):
                    print(f'additional_special_token, {i}, {j}')
        
        # Encoder & decoder config:
        config_decoder = transformers.BertConfig(
            vocab_size=len(self.tokenizer),
            num_hidden_layers=6,
            type_vocab_size=self.type_vocab_size,
        )  # BERT as it includes token_type_ids.
        config_decoder.is_decoder = True
        config_decoder.add_cross_attention = True
        encoder_ckpt_name = 'microsoft/cvt-21-384-22k'
        config_encoder = CvtWithProjectionHeadConfig.from_pretrained(
            os.path.join(self.ckpt_zoo_dir, encoder_ckpt_name),
            local_files_only=True,
            projection_size=config_decoder.hidden_size,
        )
        config = transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)

        # Encoder-to-decoder model:
        if self.warm_start_modules:
            self.encoder_decoder = LongitudinalPromptMultiCXREncoderDecoderModel(
                config=config, encoder_decoder_ckpt_name=self.multi_ckpt_name,
            )
        else:
            self.encoder_decoder = LongitudinalPromptMultiCXREncoderDecoderModel(config=config)

        # This is to get the pre-processing parameters for the checkpoint, this is not actually used for pre-processing:
        self.encoder_feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(
            os.path.join(self.ckpt_zoo_dir, encoder_ckpt_name),
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
            self.train_set = PreviousReportSubset(
                df=df.loc[df['split'] == 'train'],
                history=history.loc[history['split'] == 'train'],
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.train_transforms,
            )
            print(f'No. of training examples: {self.train_set.__len__()}.')
            print(
                f'No. of training dicom_ids & study_ids: {self.train_set.df.dicom_id.nunique()}',
                f'& {self.train_set.df.study_id.nunique()}.',
            )

        if stage == 'fit' or stage == 'validate' or stage is None:
            self.val_set = PreviousReportSubset(
                df=df.loc[df['split'] == 'validate'],
                history=history.loc[history['split'] == 'validate'],
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
            )
            print(f'No. of validation examples: {self.val_set.__len__()}.')
            print(
                f'No. of validation dicom_ids & study_ids: {self.val_set.df.dicom_id.nunique()}',
                f'& {self.val_set.df.study_id.nunique()}.',
            )

        if stage == 'test' or stage is None:
            self.test_set = PreviousReportSubset(
                df=df.loc[df['split'] == 'test'],
                history=history.loc[history['split'] == 'test'],
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
            )
            print('No. of test examples: {}.'.format(self.test_set.__len__()))
            print(
                f'No. of test dicom_ids & study_ids: {self.test_set.df.dicom_id.nunique()}',
                f'& {self.test_set.df.study_id.nunique()}.',
            )

    def forward(
            self, 
            images, 
            decoder_input_ids, 
            decoder_attention_mask, 
            decoder_token_type_ids,
            decoder_position_ids,
        ):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """

        # Teacher forcing; labels are given as input:
        outputs = self.encoder_decoder(
            pixel_values=images,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_token_type_ids=decoder_token_type_ids,
            decoder_position_ids=decoder_position_ids,
            return_dict=True,
        )

        return outputs.logits

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """

        # Tokenize report:
        tokenized = self.encoder_decoder.tokenize_report_teacher_forcing(batch['findings'], batch['impression'], self.tokenizer, self.decoder_max_len)

        # Tokenize prompt:
        prompt = self.encoder_decoder.tokenize_prompt(batch['previous_findings'], batch['previous_impression'], self.tokenizer, self.decoder_max_len)

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
        token_type_ids = self.encoder_decoder.token_ids_to_token_type_ids(
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

    def validation_step(self, batch, batch_idx):
        """
        Validation step.

        Argument/s:
            batch - mini-batch from the validation set DataLoader.
            batch_idx - batch idx of each example in the mini-batch.
        """

        # Tokenize prompt:
        prompt = self.encoder_decoder.tokenize_prompt(
            batch['previous_findings'], 
            batch['previous_impression'], 
            self.tokenizer, 
            self.decoder_max_len,  
            add_bos_token_id=True,
        )

        # Greedy search:
        output_ids = self.encoder_decoder.generate(
            pixel_values=batch['images'],
            decoder_input_ids=prompt['input_ids'],
            special_token_ids=[
                self.tokenizer.additional_special_tokens_ids[
                    self.tokenizer.additional_special_tokens.index('[PMT-SEP]')
                ],
                self.tokenizer.bos_token_id,
                self.tokenizer.sep_token_id,
            ],            
            max_length=self.decoder_max_len + prompt['input_ids'].shape[1],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']

        # An update to generate() now prepends bos_token_id to each sequence if it does not exist at the start of the input: 
        #   https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/generation/utils.py#L699C13-L699C30
        # Hence, we remove the prepended bos_token_id from each sequence if it is there:
        if torch.all(output_ids[:, 0] == 1):
            output_ids = output_ids[:, 1:]

        # Findings and impression sections (exclude previous impression section):
        _, findings, impression = self.encoder_decoder.split_and_decode_sections(
            output_ids,
            [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )

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
        prompt = self.encoder_decoder.tokenize_prompt(
            batch['previous_findings'], 
            batch['previous_impression'], 
            self.tokenizer, 
            self.decoder_max_len, 
            add_bos_token_id=True,
        )

        # Beam search:
        output_ids = self.encoder_decoder.generate(
            pixel_values=batch['images'],
            decoder_input_ids=prompt['input_ids'],
            special_token_ids=[
                self.tokenizer.additional_special_tokens_ids[
                    self.tokenizer.additional_special_tokens.index('[PMT-SEP]')
                ],
                self.tokenizer.bos_token_id,
                self.tokenizer.sep_token_id,
            ],            
            max_length=self.decoder_max_len + prompt['input_ids'].shape[1],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.pad_token_id,
            num_beams=self.num_test_beams,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']

        # An update to generate() now prepends bos_token_id to each sequence if it does not exist at the start of the input: 
        #   https://github.com/huggingface/transformers/blob/d533465150532b0c5de167b574e59f64c68b1154/src/transformers/generation/utils.py#L699C13-L699C30
        # Hence, we remove the prepended bos_token_id from each sequence if it is there:
        if torch.all(output_ids[:, 0] == 1):
            output_ids = output_ids[:, 1:]

        # Log report token identifier:
        self.test_report_ids_logger.update(output_ids, study_ids=batch['study_ids'])

        # Findings and impression sections (exclude previous impression section):
        _, findings, impression = self.encoder_decoder.split_and_decode_sections(
            output_ids,
            [self.tokenizer.bos_token_id, self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer
        )

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


