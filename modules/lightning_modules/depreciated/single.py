import os
import types
from typing import Optional

import pandas as pd
import torch
import torch.nn.functional as F
import transformers
from lightning.pytorch import LightningModule
from torch.utils.data import DataLoader
from torchvision import transforms

from task.mimic_cxr.datasets.dicom_id import DICOMIDSubset
from task.mimic_cxr.tools.metrics.chexbert import CheXbertClassificationMetrics
from task.mimic_cxr.tools.metrics.coco import COCONLGMetricsMIMICCXR
from task.mimic_cxr.tools.metrics.cxr_bert import CXRBERT
from task.mimic_cxr.tools.metrics.fact_ent import FactENT
from task.mimic_cxr.tools.metrics.fact_entnli import FactENTNLI
from task.mimic_cxr.tools.metrics.radgraph import RadGraph
from task.mimic_cxr.tools.metrics.report_ids_logger import \
    ReportIdentifiersLogger
from task.mimic_cxr.tools.metrics.report_logger import ReportLogger


class SingleCXR(LightningModule):
    """
    Single-CXR model.
    """
    def __init__(
            self,
            exp_dir_trial: str,
            dataset_dir: str,
            ckpt_zoo_dir: str,
            image_dir: str,
            module_load_apptainer: str,
            mbatch_size: int,
            decoder_max_len: int,
            lr: float,
            num_test_beams: int,
            max_images_per_study: int,
            sections_to_evaluate: list = ['report'],
            generation_config: dict = {},
            type_vocab_size: int = 2,
            max_past_studies: Optional[int] = None,
            prefetch_factor: int = 5,
            num_workers: int = 0,
            save_test_saliency_maps: bool = False,
            fact_entnli_mbatch_size: int = 1,
            accumulate_over_dicoms: bool = True,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.exp_dir_trial = exp_dir_trial
        self.dataset_dir = dataset_dir
        self.ckpt_zoo_dir = ckpt_zoo_dir
        self.image_dir = image_dir
        self.module_load_apptainer = module_load_apptainer
        self.mbatch_size = mbatch_size
        self.decoder_max_len = decoder_max_len
        self.lr = lr
        self.num_test_beams = num_test_beams
        self.max_images_per_study = max_images_per_study
        self.sections_to_evaluate = sections_to_evaluate
        self.generation_config = generation_config
        self.type_vocab_size = type_vocab_size
        self.max_past_studies = max_past_studies
        self.prefetch_factor = prefetch_factor
        self.num_workers = num_workers
        self.save_test_saliency_maps = save_test_saliency_maps
        self.image_dir = image_dir
        self.fact_entnli_mbatch_size = fact_entnli_mbatch_size
        self.accumulate_over_dicoms = accumulate_over_dicoms

        # Paths:
        self.merged_csv_path = os.path.join(self.dataset_dir, 'mimic_cxr_merged', 'splits_reports_metadata.csv')
        self.tokenizer_dir =  os.path.join(self.ckpt_zoo_dir, 'mimic-cxr-tokenizers', 'bpe_prompt')
        self.mimic_cxr_dir = os.path.join(self.dataset_dir, 'mimic_cxr_jpg', 'physionet.org', 'files', 'mimic-cxr-jpg', '2.0.0', 'files')

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
        
        # FactENT:
        # for i in self.sections_to_evaluate:
        #     self.test_metrics.append(f'test_{i}_fact_ent')
        #     setattr(
        #         self,
        #         self.test_metrics[-1],
        #         FactENT(
        #             exp_dir=self.exp_dir_trial,
        #             split=f'test_{i}',
        #             accumulate_over_dicoms=self.accumulate_over_dicoms,
        #         ),
        #     )
        
        # # FactENTNLI:
        # for i in self.sections_to_evaluate:
        #     self.test_metrics.append(f'test_{i}_fact_entnli')
        #     setattr(
        #         self,
        #         self.test_metrics[-1],
        #         FactENTNLI(
        #             image_dir=self.image_dir,
        #             exp_dir=self.exp_dir_trial,
        #             split=f'test_{i}',
        #             accumulate_over_dicoms=self.accumulate_over_dicoms,
        #             module_load_apptainer=self.module_load_apptainer,
        #             mbatch_size=self.fact_entnli_mbatch_size,
        #             num_workers=self.num_workers,
        #         ),
        #     )
        
        # RadGraph:
        for i in self.sections_to_evaluate:
            self.test_metrics.append(f'test_{i}_radgraph')
            setattr(
                self,
                self.test_metrics[-1],
                RadGraph(
                    image_dir=self.image_dir,
                    exp_dir=self.exp_dir_trial,
                    split=f'test_{i}',
                    accumulate_over_dicoms=self.accumulate_over_dicoms,
                    module_load_apptainer=self.module_load_apptainer,
                ),
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

        # Report logging
        self.val_report_logger = ReportLogger(
            exp_dir=self.exp_dir_trial, split='val_reports', track_dicom_id=self.accumulate_over_dicoms,
        )
        self.test_report_logger = ReportLogger(
            exp_dir=self.exp_dir_trial, split='test_reports', track_dicom_id=self.accumulate_over_dicoms,
        )
        self.test_report_ids_logger = ReportIdentifiersLogger(
            exp_dir=self.exp_dir_trial, split='test_report_ids', track_dicom_id=self.accumulate_over_dicoms,
        )

        # Initialise modules:
        self.init_modules()

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


    def encoder_forward(self, images):
        """
        Encoder forward propagation.

        Argument/s:
            images - a mini-batch of images.

        Returns:
            encoder_outputs - transformers.modeling_outputs.ModelOutput.
        """
        image_features = self.encoder(images).last_hidden_state
        image_features = torch.permute(torch.flatten(image_features, 2), [0, 2, 1])

        # https://github.com/huggingface/transformers/blob/b487096b02307cd6e0f132b676cdcc7255fe8e74/src/transformers/models/vit/modeling_vit.py#L569
        image_features = self.last_hidden_state_layer_norm(image_features)

        image_features = self.encoder_projection(image_features)

        encoder_outputs = transformers.modeling_outputs.BaseModelOutput(last_hidden_state=image_features)
        return encoder_outputs

    def forward(self, images, decoder_input_ids, decoder_attention_mask, decoder_token_type_ids):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        encoder_outputs = self.encoder_forward(images)

        # Teacher forcing: labels are given as input:
        outputs = self.encoder_decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_token_type_ids=decoder_token_type_ids,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )

        return outputs.logits

    def generate(self, num_beams: int, images: torch.Tensor):
        """
        Autoregressively generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            images - images for each study.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        encoder_outputs = self.encoder_forward(images)

        outputs = self.encoder_decoder.generate(
            special_token_ids=[self.tokenizer.sep_token_id],
            max_length=self.decoder_max_len,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            num_beams=num_beams,
            return_dict_in_generate=True,
            use_cache=True,
            encoder_outputs=encoder_outputs,
            **self.generation_config,
        )

        return outputs['sequences']

    def training_step(self, batch, batch_idx):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#training-step
        """

        # Tokenize report:
        tokenized = self.tokenize_report_teacher_forcing(batch['findings'], batch['impression'])
        token_type_ids = self.token_ids_to_token_type_ids(tokenized['decoder_input_ids'], [self.tokenizer.sep_token_id])

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
        output_ids = self.generate(1, batch['images'])

        # Findings and impression sections:
        findings, impression = self.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
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
        output_ids = self.generate(self.num_test_beams, batch['images'])

        # Log report token identifier:
        self.test_report_ids_logger.update(output_ids, dicom_ids=batch['dicom_ids'], study_ids=batch['study_ids'])

        # Findings and impression sections:
        findings, impression = self.split_and_decode_sections(
            output_ids,
            [self.tokenizer.sep_token_id, self.tokenizer.eos_token_id],
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

    def tokenize_report_teacher_forcing(self, findings: str, impression: str):
        """
        Tokenize the reports and creates the inputs and targets for teacher forcing.

        Argument/s:
            findings - findings section.
            impression - impression section.
            return_token_type_ids - return the token type identifiers.

        Returns:
            decoder_input_ids - the token identifiers for the input of the decoder.
            decoder_attention_mask - the attention mask for the decoder_input_ids.
            label_ids - the label token identifiers for the decoder.
        """

        # Prepare the sections for the tokenizer by placing special tokens between each section:
        report = [f'{self.tokenizer.bos_token}{i}{self.tokenizer.sep_token}{j}{self.tokenizer.eos_token}' for i, j in
                  zip(findings, impression)]

        # Tokenize the report:
        tokenized = self.tokenizer(
            report,
            padding='longest',
            truncation=True,
            max_length=self.decoder_max_len + 1,  # +1 to account for the bias between input and target.
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

    def split_and_decode_sections(self, token_ids, special_token_ids):
        """
        Split the token identifiers into sections, then convert the token identifiers into strings.

        Argument/s:
            token_ids - token identifiers.
            special_token_ids - special token identifiers that indicate the end of each section.

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
                section_string = self.tokenizer.decode(section_token_ids, skip_special_tokens=True)

                sections[j].append(section_string)

        return tuple(sections.values())

    @staticmethod
    def prepare_inputs_for_generation(
        self,
        input_ids,
        special_token_ids,
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
        decoder_attention_mask = decoder_inputs['attention_mask'] if 'attention_mask' in decoder_inputs else None

        if not past_key_values:
            token_type_ids = SingleCXR.token_ids_to_token_type_ids(input_ids, special_token_ids)
        else:
            token_type_ids = SingleCXR.token_ids_to_token_type_ids_past(input_ids, special_token_ids)

        input_dict = {
            'attention_mask': attention_mask,
            'decoder_attention_mask': decoder_attention_mask,
            'decoder_input_ids': decoder_inputs['input_ids'],
            'decoder_token_type_ids': token_type_ids,
            'encoder_outputs': encoder_outputs,
            'past_key_values': decoder_inputs['past_key_values'],
            'use_cache': use_cache,
        }
        return input_dict

    @staticmethod
    def token_ids_to_token_type_ids(token_ids, special_token_ids, token_type_id_sections=None):
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

    @staticmethod
    def token_ids_to_token_type_ids_past(token_ids, special_token_ids, token_type_id_sections=None):
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
