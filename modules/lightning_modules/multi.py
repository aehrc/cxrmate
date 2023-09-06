import os
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers.modeling_outputs import BaseModelOutput

from data.study_id import StudyIDSubset
from modules.lightning_modules.single import SingleCXR
from modules.transformers.multi_model.modelling_multi import (
    CvtWithProjectionHeadConfig, MultiCvtWithProjectionHead,
    MultiCXREncoderDecoderModel)


class MultiCXR(SingleCXR):
    """
    Multi-image CXR report generation model.
    """
    def __init__(self, **kwargs):
        kwargs['accumulate_over_dicoms'] = False
        super().__init__(**kwargs)

    def init_modules(self):
        """
        Initialise torch.nn.Modules.
        """

        encoder_decoder_ckpt_name = 'aehrc/cxrmate-multi-tf'

        # Decoder tokenizer:
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(encoder_decoder_ckpt_name)
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
        config_encoder = CvtWithProjectionHeadConfig.from_pretrained(encoder_ckpt_name, projection_size=config_decoder.hidden_size)

        # Encoder-to-decoder model:
        if self.warm_start_modules:
            encoder = MultiCvtWithProjectionHead.from_pretrained(encoder_ckpt_name, config=config_encoder)
            decoder = transformers.BertLMHeadModel(config=config_decoder)
            self.encoder_decoder = MultiCXREncoderDecoderModel(encoder=encoder, decoder=decoder)
        else:
            config = transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
            self.encoder_decoder = MultiCXREncoderDecoderModel(config=config)

        # This is to get the pre-processing parameters for the checkpoint, this is not actually used for pre-processing:
        self.encoder_feature_extractor = transformers.AutoFeatureExtractor.from_pretrained(encoder_ckpt_name)

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

        # Drop studies that don't have a findings or impression section:
        df = df.dropna(subset=['findings', 'impression'], how='any')

        # Drop studies that have more than the maximum number of DICOMs per study:
        df = df[df.study_id.map(df.study_id.value_counts()) <= self.max_images_per_study]

        if stage == 'fit' or stage is None:
            self.train_set = StudyIDSubset(
                df=df.loc[df['split'] == 'train'],
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.train_transforms,
            )
            print(f'No. of training examples: {self.train_set.__len__()}.')
            print(
                f'No. of training dicom_ids & study_ids: {self.train_set.df.dicom_id.nunique()}',
                f'& {self.train_set.df.study_id.nunique()}.',
            )

        if stage == 'fit' or stage == 'validate' or stage is None:
            self.val_set = StudyIDSubset(
                df=df.loc[df['split'] == 'validate'],
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
            )
            print(f'No. of validation examples: {self.val_set.__len__()}.')
            print(
                f'No. of validation dicom_ids & study_ids: {self.val_set.df.dicom_id.nunique()}',
                f'& {self.val_set.df.study_id.nunique()}.',
            )

        if stage == 'test' or stage is None:
            self.test_set = StudyIDSubset(
                df=df.loc[df['split'] == 'test'],
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
            )
            print('No. of test examples: {}.'.format(self.test_set.__len__()))
            print(
                f'No. of test dicom_ids & study_ids: {self.test_set.df.dicom_id.nunique()}',
                f'& {self.test_set.df.study_id.nunique()}.',
            )

    @staticmethod
    def collate_fn(batch):
        """
        https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        """

        batch = {j: [i[j] for i in batch] for j in batch[0]}
        batch['images'] = torch.nn.utils.rnn.pad_sequence(batch['images'], batch_first=True, padding_value=0.0)

        return batch

    def forward(self, images, decoder_input_ids, decoder_attention_mask, decoder_token_type_ids):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """

        # Teacher forcing: labels are given as input:
        outputs = self.encoder_decoder(
            pixel_values=images,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_token_type_ids=decoder_token_type_ids,
            return_dict=True,
        )

        return outputs.logits

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """

        # Tokenize report:
        tokenized = self.encoder_decoder.tokenize_report_teacher_forcing(
            batch['findings'], batch['impression'], self.tokenizer, self.decoder_max_len,
        )
        token_type_ids = self.encoder_decoder.token_ids_to_token_type_ids(tokenized['decoder_input_ids'], [self.tokenizer.sep_token_id])

        # Inference:
        y_hat = self(            
            batch['images'], 
            tokenized['decoder_input_ids'],
            tokenized['decoder_attention_mask'], 
            token_type_ids,
        )

        # Loss:
        loss = F.cross_entropy(
            y_hat.permute([0, 2, 1]), tokenized['label_ids'], ignore_index=self.tokenizer.pad_token_id,
        )

        # Logging:
        self.log_dict({'train_loss': loss}, on_step=True, on_epoch=True, batch_size=batch['images'].size()[0])

        # Update and log scores for each validation metric:
        return loss

    def validation_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#validation-step
        """

        # Greedy search:
        output_ids = self.encoder_decoder.generate(
            pixel_values=batch['images'],
            special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, 
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=1,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']

        # Findings and impression sections:
        findings, impression = self.encoder_decoder.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
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
            
    def test_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-step
        """

        # Beam search:
        output_ids = self.encoder_decoder.generate(
            pixel_values=batch['images'],
            special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=self.num_test_beams,
            return_dict_in_generate=True,
            use_cache=True,
        )['sequences']

        # Log report token identifier:
        self.test_report_ids_logger.update(output_ids, study_ids=batch['study_ids'])

        # Findings and impression sections:
        findings, impression = self.encoder_decoder.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
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
            