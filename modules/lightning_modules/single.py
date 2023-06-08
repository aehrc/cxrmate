import os
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch import LightningModule
from tools.metrics.chexbert import CheXbertClassificationMetrics
from tools.metrics.coco import COCONLGMetricsMIMICCXR
from tools.metrics.cxr_bert import CXRBERT
from tools.metrics.report_ids_logger import \
    ReportTokenIdentifiersLogger
from tools.metrics.report_logger import ReportLogger
from torch.utils.data import DataLoader
from torchvision import transforms

from data.dicom_id import DICOMIDSubset
from modules.transformers.single_model.modelling_single import (
    CvtWithProjectionHead, CvtWithProjectionHeadConfig,
    SingleCXREncoderDecoderModel)


class SingleCXR(LightningModule):
    """
    Single-CXR model.
    """
    def __init__(
            self,
            warm_start_modules: bool,
            exp_dir_trial: str,
            dataset_dir: str,
            ckpt_zoo_dir: Optional[str] = None,
            mbatch_size: Optional[int] = None,
            decoder_max_len: Optional[int] = None,
            lr: Optional[float] = None,
            num_test_beams: Optional[int] = None,
            max_images_per_study: Optional[int] = None,
            sections_to_evaluate: list = ['report'],
            type_vocab_size: int = 2,
            prefetch_factor: int = 5,
            num_workers: int = 0,
            accumulate_over_dicoms: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.warm_start_modules = warm_start_modules
        self.exp_dir_trial = exp_dir_trial
        self.dataset_dir = dataset_dir
        self.ckpt_zoo_dir = ckpt_zoo_dir
        self.mbatch_size = mbatch_size
        self.decoder_max_len = decoder_max_len
        self.lr = lr
        self.num_test_beams = num_test_beams
        self.max_images_per_study = max_images_per_study
        self.sections_to_evaluate = sections_to_evaluate
        self.type_vocab_size = type_vocab_size
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.accumulate_over_dicoms = accumulate_over_dicoms

        # Paths:
        self.merged_csv_path = os.path.join(self.dataset_dir, 'mimic_cxr_merged', 'splits_reports_metadata.csv')
        self.tokenizer_dir =  os.path.join(self.ckpt_zoo_dir, 'mimic-cxr-tokenizers', 'bpe_prompt')
        self.mimic_cxr_dir = os.path.join(self.dataset_dir, 'physionet.org', 'files', 'mimic-cxr-jpg', '2.0.0', 'files')

        """
        Evaluation metrics
        
        These need to be defined correctly in order for them to be placed on the correct device:
        https://torchmetrics.readthedocs.io/en/stable/pages/lightning.html#torchmetrics-in-pytorch-lightning
        """
        self.val_metrics, self.test_metrics = [], []

        # COCO NLG metrics:
        for i in self.sections_to_evaluate:
            self.val_metrics.append(f'val_{i}_nlg')
            setattr(
                self,
                self.val_metrics[-1],
                COCONLGMetricsMIMICCXR(
                    split=f'val_{i}',
                    metrics=['bleu', 'cider', 'rouge'],
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )
        
        # Cannot deepcopy either SPICE or METEOR:
        for i in self.sections_to_evaluate:
            self.test_metrics.append(f'test_{i}_nlg')
            setattr(
                self,
                self.test_metrics[-1],
                COCONLGMetricsMIMICCXR(
                    split=f'test_{i}',
                    metrics=['bleu', 'cider', 'rouge', 'meteor'],
                    exp_dir=self.exp_dir_trial,
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )
        
        # CheXbert metrics:
        for i in self.sections_to_evaluate:
            self.val_metrics.append(f'val_{i}_chexbert')
            setattr(
                self,
                self.val_metrics[-1],
                CheXbertClassificationMetrics(
                    bert_path='bert-base-uncased',
                    checkpoint_path='stanford/chexbert/chexbert.pth',
                    ckpt_dir=self.ckpt_zoo_dir,
                    mbatch_size=self.mbatch_size,
                    exp_dir=self.exp_dir_trial,
                    split=f'val_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                )
            )
        for i in self.sections_to_evaluate:
            self.test_metrics.append(f'test_{i}_chexbert')
            setattr(
                self,
                self.test_metrics[-1],
                CheXbertClassificationMetrics(
                    bert_path='bert-base-uncased',
                    checkpoint_path='stanford/chexbert/chexbert.pth',
                    ckpt_dir=self.ckpt_zoo_dir,
                    mbatch_size=self.mbatch_size,
                    exp_dir=self.exp_dir_trial,
                    split=f'test_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                )
            )
        
        # CXR-BERT:
        for i in self.sections_to_evaluate:
            self.test_metrics.append(f'test_{i}_cxr-bert')
            setattr(
                self,
                self.test_metrics[-1],
                CXRBERT(
                    ckpt_dir=self.ckpt_zoo_dir,
                    mbatch_size=self.mbatch_size,
                    exp_dir=self.exp_dir_trial,
                    split=f'test_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                ),
            )

        # Report logging:
        self.val_report_logger = ReportLogger(
            exp_dir=self.exp_dir_trial, split='val_reports', track_dicom_id=self.accumulate_over_dicoms,
        )
        self.test_report_logger = ReportLogger(
            exp_dir=self.exp_dir_trial, split='test_reports', track_dicom_id=self.accumulate_over_dicoms,
        )
        self.test_report_ids_logger = ReportTokenIdentifiersLogger(
            exp_dir=self.exp_dir_trial, split='test_report_ids', track_dicom_id=self.accumulate_over_dicoms,
        )

        # Initialise modules:
        self.init_modules()

    def init_modules(self):
        """
        Initialise torch.nn.Modules.
        """

        encoder_decoder_ckpt_name = 'aehrc/mimic-cxr-report-gen-single'

        # Decoder tokenizer:
        self.tokenizer = transformers.PreTrainedTokenizerFast.from_pretrained(encoder_decoder_ckpt_name, cache_dir=os.path.join(self.ckpt_zoo_dir))
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
            self.encoder_decoder = SingleCXREncoderDecoderModel(encoder=encoder, decoder=decoder)
        else:
            config = transformers.VisionEncoderDecoderConfig.from_encoder_decoder_configs(config_encoder, config_decoder)
            self.encoder_decoder = SingleCXREncoderDecoderModel(config=config)

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
            self.train_set = DICOMIDSubset(
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
            self.val_set = DICOMIDSubset(
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
            self.test_set = DICOMIDSubset(
                df=df.loc[df['split'] == 'test'],
                dataset_dir=self.mimic_cxr_dir,
                transforms=self.test_transforms,
            )
            print('No. of test examples: {}.'.format(self.test_set.__len__()))
            print(
                f'No. of test dicom_ids & study_ids: {self.test_set.df.dicom_id.nunique()}',
                f'& {self.test_set.df.study_id.nunique()}.',
            )

    def train_dataloader(self, shuffle=True):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#train-dataloader
        """
        return DataLoader(
            self.train_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#val-dataloader
        """
        return DataLoader(
            self.val_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#test-dataloader
        """
        return DataLoader(
            self.test_set,
            batch_size=self.mbatch_size,
            num_workers=self.num_workers,
            shuffle=False,
            prefetch_factor=self.prefetch_factor,
            collate_fn=self.collate_fn,
        )

    @staticmethod
    def collate_fn(batch):
        """
        https://pytorch.org/docs/stable/data.html#working-with-collate-fn
        """

        batch = {j: [i[j] for i in batch] for j in batch[0]}
        batch['images'] = torch.stack(batch['images'])

        return batch

    def configure_optimizers(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#configure-optimizers
        """
        optimiser = {'optimizer': torch.optim.AdamW(self.parameters(), lr=self.lr)}
        return optimiser

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
        tokenized = self.encoder_decoder.tokenize_report_teacher_forcing(batch['findings'], batch['impression'], self.tokenizer, self.decoder_max_len)
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
            num_beams=2,
            return_dict_in_generate=True,
            use_cache=True,
        )

        # Findings and impression sections:
        findings, impression = self.encoder_decoder.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )

        # Log reports:
        self.val_report_logger.update(findings, impression, dicom_ids=batch['dicom_ids'], study_ids=batch['study_ids'])

        # Evaluate:
        for i in self.val_metrics:
            if 'findings' in i:
                getattr(self, i).update(
                    findings, [[j] for j in batch['findings']], dicom_ids=batch['dicom_ids'], study_ids=batch['study_ids'],
                )
            elif 'impression' in i:
                getattr(self, i).update(
                    impression, [[j] for j in batch['impression']], dicom_ids=batch['dicom_ids'], study_ids=batch['study_ids'],
                )
            elif 'report' in i:
                getattr(self, i).update(
                    [f'{i} {j}' for i, j in zip(findings, impression)],
                    [[f'{i} {j}'] for i, j in zip(batch['findings'], batch['impression'])],
                    dicom_ids=batch['dicom_ids'],
                    study_ids=batch['study_ids'],
                )
            else:
                raise ValueError(f'{i} must contain findings, impression, or report')

    def on_validation_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-validation-epoch-end
        """

        # Save reports:
        self.val_report_logger.compute(self.current_epoch)
        self.val_report_logger.reset()

        scores = {}
        for i in self.val_metrics:
            output = getattr(self, i).compute(self.current_epoch)
            if isinstance(output, dict):
                for k, v in output.items():
                    scores.update({f'{i}_{k}': v})
            else:
                scores.update({f'{i}': output})

        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
        [getattr(self, i).reset() for i in self.val_metrics]

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
        )

        # Log report token identifier:
        self.test_report_ids_logger.update(output_ids, dicom_ids=batch['dicom_ids'], study_ids=batch['study_ids'])

        # Findings and impression sections:
        findings, impression = self.encoder_decoder.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
            self.tokenizer,
        )

        # Log reports:
        self.test_report_logger.update(findings, impression, dicom_ids=batch['dicom_ids'], study_ids=batch['study_ids'])

        # Evaluate:
        for i in self.test_metrics:
            if 'findings' in i:
                getattr(self, i).update(
                    findings, [[j] for j in batch['findings']], dicom_ids=batch['dicom_ids'], study_ids=batch['study_ids'],
                )
            elif 'impression' in i:
                getattr(self, i).update(
                    impression, [[j] for j in batch['impression']], dicom_ids=batch['dicom_ids'], study_ids=batch['study_ids'],
                )
            elif 'report' in i:
                getattr(self, i).update(
                    [f'{i} {j}' for i, j in zip(findings, impression)],
                    [[f'{i} {j}'] for i, j in zip(batch['findings'], batch['impression'])],
                    dicom_ids=batch['dicom_ids'],
                    study_ids=batch['study_ids'],
                )
            else:
                raise ValueError(f'{i} must contain findings, impression, or report')

    def on_test_epoch_end(self):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#on-test-epoch-end
        """

        # Save reports:
        self.test_report_logger.compute(self.current_epoch)
        self.test_report_logger.reset()
        self.test_report_ids_logger.compute(self.current_epoch)
        self.test_report_ids_logger.reset()

        scores = {}
        for i in self.test_metrics:
            output = getattr(self, i).compute(self.current_epoch)
            if isinstance(output, dict):
                for k, v in output.items():
                    scores.update({f'{i}_{k}': v})
            else:
                scores.update({f'{i}': output})
        self.log_dict({f'{k}': v for k, v in scores.items()}, on_step=False, on_epoch=True)
        [getattr(self, i).reset() for i in self.test_metrics]
