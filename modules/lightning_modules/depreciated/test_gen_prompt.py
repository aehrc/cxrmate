

import pandas as pd
from torch.utils.data import DataLoader

from task.mimic_cxr.datasets.prompt import PreviousReportSubset
from task.mimic_cxr.model.report_gen.any.prompt import GTPrompt


class GeneratedPrompt(GTPrompt):

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
