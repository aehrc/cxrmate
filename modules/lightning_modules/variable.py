import os
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from torchvision import transforms
from transformers.modeling_outputs import BaseModelOutput

from data.study_id import StudyIDSubset
from modules.lightning_modules.single import SingleCXR
from modules.transformers.variable_model.modelling_variable import (
    CvtWithProjectionHead, CvtWithProjectionHeadConfig,
    VariableCXREncoderDecoderModel)


class VariableCXR(SingleCXR):
    """
    Variable-CXR model.
    """
    def __init__(self, **kwargs):
        kwargs['accumulate_over_dicoms'] = False
        super().__init__(**kwargs)

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

        # Encoder & decoder config:
        config_decoder = transformers.BertConfig(
            vocab_size=len(self.tokenizer),
            num_hidden_layers=6,
            type_vocab_size=self.type_vocab_size,
        )  # BERT as it includes token_type_ids.
        encoder_ckpt_name = 'microsoft/cvt-21-384-22k'
        config_encoder = CvtWithProjectionHeadConfig.from_pretrained(
            os.path.join(self.ckpt_zoo_dir, encoder_ckpt_name),
            local_files_only=True,
            projection_size=config_decoder.hidden_size,
        )

        # Encoder-to-decoder model:
        if self.warm_start_modules:
            encoder = CvtWithProjectionHead.from_pretrained(
                os.path.join(self.ckpt_zoo_dir, encoder_ckpt_name),
                local_files_only=True,
                config=config_encoder,
            )
            decoder = transformers.BertLMHeadModel(config=config_decoder)
            self.encoder_decoder = VariableCXREncoderDecoderModel(encoder=encoder, decoder=decoder)
        else:
            config = transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
            self.encoder_decoder = VariableCXREncoderDecoderModel(config=config)

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
        batch['dicom_study_ids'] = [j for i in batch['dicom_study_ids'] for j in i]
        batch['images'] = torch.cat(batch['images'], dim=0)

        return batch

    def forward(self, images, dicom_study_ids, decoder_input_ids, decoder_attention_mask, decoder_token_type_ids):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """

        # Teacher forcing: labels are given as input:
        outputs = self.encoder_decoder(
            pixel_values=images,
            dicom_study_ids=dicom_study_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_token_type_ids=decoder_token_type_ids,
            return_dict=True,
        )

        return outputs.logits

    # def encoder_forward(self, images, dicom_study_ids):
    #     """
    #     Encoder forward propagation.

    #     Argument/s:
    #         images - a mini-batch of images.
    #         dicom_study_ids - DICOM study ID.

    #     Returns:
    #         encoder_outputs - transformers.modeling_outputs.ModelOutput.
    #     """
    #     image_features = self.encoder(images).last_hidden_state
    #     image_features = torch.permute(torch.flatten(image_features, 2), [0, 2, 1])

    #     # https://github.com/huggingface/transformers/blob/b487096b02307cd6e0f132b676cdcc7255fe8e74/src/transformers/models/vit/modeling_vit.py#L569
    #     image_features = self.last_hidden_state_layer_norm(image_features)

    #     image_features = self.encoder_projection(image_features)

    #     mbatch_size = len(set(dicom_study_ids))
    #     max_images = dicom_study_ids.count(max(dicom_study_ids, key=dicom_study_ids.count))
    #     feature_size = image_features.shape[-1]
    #     spatial_positions = image_features.shape[-2]

    #     attention_mask = torch.zeros(mbatch_size, max_images * spatial_positions).to(self.device)
    #     reshaped_image_features = torch.zeros(
    #         mbatch_size, max_images * spatial_positions, feature_size, dtype=image_features.dtype,
    #     ).to(self.device)

    #     #  There has to be a better way to do the following:
    #     row_count, column_count = 0, 0
    #     previous = dicom_study_ids[0]
    #     for i, j in enumerate(dicom_study_ids):
    #         if j != previous:
    #             row_count += 1
    #             column_count = 0
    #         attention_mask[row_count, column_count:column_count + spatial_positions] = 1.0
    #         reshaped_image_features[row_count, column_count:column_count + spatial_positions] = image_features[i]
    #         column_count += spatial_positions
    #         previous = j

    #     encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=reshaped_image_features)

    #     return encoder_outputs, attention_mask

    # def forward(self, images, dicom_study_ids, decoder_input_ids, decoder_attention_mask, decoder_token_type_ids):
    #     """
    #     https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
    #     """
    #     encoder_outputs, attention_mask = self.encoder_forward(images, dicom_study_ids)

    #     # Teacher forcing: labels are given as input:
    #     outputs = self.encoder_decoder(
    #         decoder_input_ids=decoder_input_ids,
    #         decoder_attention_mask=decoder_attention_mask,
    #         decoder_token_type_ids=decoder_token_type_ids,
    #         attention_mask=attention_mask,
    #         encoder_outputs=encoder_outputs,
    #         return_dict=True,
    #     )

    #     return outputs.logits

    # def generate(
    #     self,
    #     num_beams: int, 
    #     dicom_study_ids: List[int], 
    #     images: Optional[torch.Tensor] = None, 
    #     encoder_outputs: Optional[BaseModelOutput] = None,
    #     attention_mask: Optional[torch.Tensor] = None,
    # ):
    #     """
    #     Autoregressively generate a prediction.

    #     Argument/s:
    #         num_beams - number of considered beams for the search (one beam is a greedy search).
    #         dicom_study_ids - study identifier of each DICOM.
    #         prompt_ids - token identifiers of the previous impression section to prompt the next report.
    #         images - images for each study.
    #         encoder_outputs - outputs of the encoder.
    #         attention_mask - attention mask for the cross-attention.

    #     Returns:
    #         Indices of the tokens for the predicted sequence.
    #     """

    #     # Encoder outputs for cross-attention:
    #     if encoder_outputs is None:
    #         encoder_outputs, attention_mask = self.encoder_forward(images, dicom_study_ids)

    #     outputs = self.encoder_decoder.generate(
    #         special_token_ids=[self.tokenizer.sep_token_id],
    #         max_length=self.decoder_max_len,
    #         bos_token_id=self.tokenizer.bos_token_id,
    #         eos_token_id=self.tokenizer.eos_token_id,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         num_beams=num_beams,
    #         return_dict_in_generate=True,
    #         use_cache=True,
    #         attention_mask=attention_mask,
    #         encoder_outputs=encoder_outputs,
    #         **self.generation_config,
    #     )

    #     return outputs['sequences']

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
            batch['dicom_study_ids'],
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
        output_ids = self.generate(
            num_beams=1, 
            dicom_study_ids=batch['dicom_study_ids'], 
            images=batch['images'], 
        )

        # Findings and impression sections:
        findings, impression = self.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
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
        output_ids = self.generate(
            num_beams=self.num_test_beams, 
            dicom_study_ids=batch['dicom_study_ids'], 
            images=batch['images'], 
        )

        # Log report token identifier:
        self.test_report_ids_logger.update(output_ids, study_ids=batch['study_ids'])

        # Findings and impression sections:
        findings, impression = self.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
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
            