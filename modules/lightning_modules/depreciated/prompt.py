from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F
from transformers.modeling_outputs import BaseModelOutput

from task.mimic_cxr.datasets.prompt import PreviousReportSubset
from task.mimic_cxr.model.report_gen.any.single import SingleCXR
from task.mimic_cxr.model.report_gen.any.variable import VariableCXR



class GTPrompt(VariableCXR):
    """
    Prompt the decoder with the findings and impression section of the previous study.
    """
    def __init__(self, **kwargs):
        kwargs['type_vocab_size'] = 4
        super().__init__(**kwargs)

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
            dicom_study_ids, 
            decoder_input_ids, 
            decoder_attention_mask, 
            decoder_token_type_ids,
            decoder_position_ids,
        ):
        """
        https://lightning.ai/docs/pytorch/stable/common/lightning_module.html#forward
        """
        encoder_outputs, attention_mask = self.encoder_forward(images, dicom_study_ids)

        # Teacher forcing; labels are given as input:
        outputs = self.encoder_decoder(
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            decoder_token_type_ids=decoder_token_type_ids,
            decoder_position_ids=decoder_position_ids,
            attention_mask=attention_mask,
            encoder_outputs=encoder_outputs,
            return_dict=True,
        )

        return outputs.logits

    def generate(
        self,
        num_beams: int, 
        dicom_study_ids: List[int], 
        prompt_ids: torch.Tensor,
        images: Optional[torch.Tensor] = None, 
        encoder_outputs: Optional[BaseModelOutput] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        """
        Autoregressively generate a prediction.

        Argument/s:
            num_beams - number of considered beams for the search (one beam is a greedy search).
            dicom_study_ids - study identifier of each DICOM.
            prompt_ids - token identifiers of the previous impression section to prompt the next report.
            images - images for each study.
            encoder_outputs - outputs of the encoder.
            attention_mask - attention mask for the cross-attention.

        Returns:
            Indices of the tokens for the predicted sequence.
        """

        # Encoder outputs for cross-attention:
        if encoder_outputs is None:
            encoder_outputs, attention_mask = self.encoder_forward(images, dicom_study_ids)

        # Generate the report:
        outputs = self.encoder_decoder.generate(
            decoder_input_ids=prompt_ids,
            special_token_ids=[
                self.tokenizer.additional_special_tokens_ids[
                    self.tokenizer.additional_special_tokens.index('[PMT-SEP]')
                ],
                self.tokenizer.bos_token_id,
                self.tokenizer.sep_token_id,
            ],            
            max_length=self.decoder_max_len + prompt_ids.shape[1],
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            mask_token_id=self.tokenizer.pad_token_id,
            num_beams=num_beams,
            return_dict_in_generate=True,
            use_cache=True,
            attention_mask=attention_mask,
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

    def tokenize_prompt(self, previous_findings: str, previous_impression: str, add_bos_token_id: bool = False):
        """
        Tokenize the sections of the previous report to be used as a prompt.

        Argument/s:
            previous_findings - previous findings section.
            previous_impression - previous impression section.
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
            f'[PMT]{i}[PMT-SEP]{j}{self.tokenizer.bos_token}' if add_bos_token_id else f'[PMT]{i}[PMT-SEP]{j}' \
                for i, j in zip(previous_findings, previous_impression)
        ]

        # Tokenize:
        previous_sections = self.tokenizer(
            previous_sections,
            padding='longest',
            truncation=True,
            max_length=self.decoder_max_len,
            return_tensors='pt',
            return_token_type_ids=False,
            add_special_tokens=False,
        ).to(self.device)

        # Ensure BOS token identifier is at the end of the input_ids:
        if previous_sections.input_ids.shape[1] == self.decoder_max_len:
            previous_sections.input_ids[:, -1] = torch.where(
                previous_sections.attention_mask[:, -1] == 1,
                self.tokenizer.bos_token_id,
                previous_sections.input_ids[:, -1],
            ) 

        assert previous_sections.input_ids.shape[1] <= self.decoder_max_len

        return {'input_ids': previous_sections.input_ids, 'attention_mask': previous_sections.attention_mask}

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
            token_type_ids = SingleCXR.token_ids_to_token_type_ids(input_ids, special_token_ids)
        else:
            token_type_ids = SingleCXR.token_ids_to_token_type_ids_past(input_ids, special_token_ids)
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
