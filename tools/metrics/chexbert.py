from pathlib import Path
from torchmetrics import Metric
from tools.chexbert import CheXbert
import os
import pandas as pd
import time
import torch

"""
0 = blank/not mentioned
1 = positive
2 = negative
3 = uncertain
"""

PATHOLOGIES = [
    'enlarged_cardiomediastinum',
    'cardiomegaly',
    'lung_opacity',
    'lung_lesion',
    'edema',
    'consolidation',
    'pneumonia',
    'atelectasis',
    'pneumothorax',
    'pleural_effusion',
    'pleural_other',
    'fracture',
    'support_devices',
    'no_finding',
]


class CheXbertClassificationMetrics(Metric):
    """
    CheXbert classification metrics for MIMIC-CXR. If multiple reports are generated per study_id, each error type is
    summed over the dicom_ids.
    """

    def __init__(
            self, split, ckpt_dir, bert_path, checkpoint_path, mbatch_size, exp_dir, accumulate_over_dicoms,
    ):
        """
        Argument/s:
            split - dataset split.
            ckpt_dir - path to the checkpoint directory.
            bert_path - path to the Hugging Face BERT checkpoint (for the BERT configuration).
            checkpoint_path - path to the CheXbert checkpoint.
            mbatch_size - mini-batch size for CheXbert.
            exp_dir - experiment directory where outputs will be saved.
            accumulate_over_dicoms - whether to accumulate scores over the report for each DICOM for a study.
        """
        super().__init__(dist_sync_on_step=False, compute_on_step=False)

        self.split = split
        self.ckpt_dir = ckpt_dir
        self.bert_path = bert_path
        self.checkpoint_path = checkpoint_path
        self.mbatch_size = mbatch_size
        self.exp_dir = exp_dir
        self.accumulate_over_dicoms = accumulate_over_dicoms

        self.add_state('reports', default=[])

        self.save_dir = os.path.join(self.exp_dir, 'chexbert_outputs')
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def mini_batch(iterable, mbatch_size=1):
        length = len(iterable)
        for i in range(0, length, mbatch_size):
            yield iterable[i:min(i + mbatch_size, length)]

    def update(self, predictions, labels, study_ids, dicom_ids=None):
        """
        Argument/s:
            predictions - the predictions must be in the following format:

                [
                    '...',
                    '...',
                ]
            labels - the labels must be in the following format:

                [
                    ['...'],
                    ['...'],
                ]
            study_ids - list of study identifiers.
            dicom_ids - list of dicom identifiers.
        """

        assert isinstance(predictions, list), '"predictions" must be a list of strings.'
        assert all(isinstance(i, str) for i in predictions), 'Each element of "predictions" must be a string.'
        assert isinstance(labels, list), '"labels" must be a list of lists, where each sub-list has a multiple strings.'
        assert all(isinstance(i, list) for i in labels), 'Each element of "labels" must be a list of strings.'
        assert all(isinstance(j, str) for i in labels for j in i), 'each sub-list must have one or more strings.'

        if self.accumulate_over_dicoms:
            for (i_1, i_2, i_3, i_4) in zip(predictions, labels, study_ids, dicom_ids):
                self.reports.append({'prediction': i_1, 'label': i_2, 'study_id': i_3, 'dicom_id': i_4})
        else:
            for (i_1, i_2, i_3) in zip(predictions, labels, study_ids):
                self.reports.append({'prediction': i_1, 'label': i_2, 'study_id': i_3})

    def compute(self, epoch):

        chexbert = CheXbert(
            ckpt_dir=self.ckpt_dir,
            bert_path=self.bert_path,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
        ).to(self.device)

        y_hat_rows = []
        y_rows = []
        for i in self.mini_batch(self.reports, self.mbatch_size):
            y_hat = [j['prediction'] for j in i]
            y = [j['label'] for j in i]
            study_ids = [j['study_id'] for j in i]
            if self.accumulate_over_dicoms:
                dicom_ids = [j['dicom_id'] for j in i]

            # Following COCO, the labels are contained in a nested list:
            for j in y:
                assert len(j) == 1
            y = [j[0] for j in y]

            y_hat_chexbert = chexbert(list(y_hat)).tolist()
            y_chexbert = chexbert(list(y)).tolist()

            y_hat_mbatch_rows = []
            y_mbatch_rows = []
            if self.accumulate_over_dicoms:
                for i_1, i_2, i_3, i_4 in zip(dicom_ids, study_ids, y_hat_chexbert, y_chexbert):
                    y_hat_mbatch_rows.append(
                        {**{'dicom_id': i_1, 'study_id': i_2}, **{k: v for k, v in zip(PATHOLOGIES, i_3)}}
                    )
                    y_mbatch_rows.append(
                        {**{'dicom_id': i_1, 'study_id': i_2}, **{k: v for k, v in zip(PATHOLOGIES, i_4)}}
                    )
            else:
                for i_1, i_2, i_3 in zip(study_ids, y_hat_chexbert, y_chexbert):
                    y_hat_mbatch_rows.append(
                        {**{'study_id': i_1}, **{k: v for k, v in zip(PATHOLOGIES, i_2)}}
                    )
                    y_mbatch_rows.append(
                        {**{'study_id': i_1}, **{k: v for k, v in zip(PATHOLOGIES, i_3)}}
                    )

            y_hat_rows.extend(y_hat_mbatch_rows)
            y_rows.extend(y_mbatch_rows)

        # Gather if DDP
        if torch.distributed.is_initialized():
            y_hat_rows_gathered = [None] * torch.distributed.get_world_size()
            y_rows_gathered = [None] * torch.distributed.get_world_size()

            torch.distributed.all_gather_object(y_hat_rows_gathered, y_hat_rows)
            torch.distributed.all_gather_object(y_rows_gathered, y_rows)

            y_hat_rows = [j for i in y_hat_rows_gathered for j in i]
            y_rows = [j for i in y_rows_gathered for j in i]

        scores = {'y_hat': pd.DataFrame(y_hat_rows), 'y': pd.DataFrame(y_rows)}

        # Drop duplicates caused by DDP
        key = 'dicom_id' if self.accumulate_over_dicoms else 'study_id'
        scores['y_hat'] = scores['y_hat'].drop_duplicates(subset=[key])
        scores['y'] = scores['y'].drop_duplicates(subset=[key])

        def save_chexbert_outputs():
            scores['y_hat'].to_csv(
                os.path.join(
                    self.save_dir, f'{self.split}_epoch-{epoch}_y_hat_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'
                ),
                index=False,
            )
            scores['y'].to_csv(
                os.path.join(
                    self.save_dir, f'{self.split}_epoch-{epoch}_y_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'
                ),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save_chexbert_outputs()
        elif torch.distributed.get_rank() == 0:
            save_chexbert_outputs()

        # Positive is 1/positive, negative is 0/not mentioned, 2/negative, and 3/uncertain:
        scores['y_hat'][PATHOLOGIES] = (scores['y_hat'][PATHOLOGIES] == 1)
        scores['y'][PATHOLOGIES] = (scores['y'][PATHOLOGIES] == 1)

        # Create dataframes for each error type
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores[i] = scores['y'][['study_id', 'dicom_id']].copy() if self.accumulate_over_dicoms \
                else scores['y'][['study_id']].copy()

        # Calculate errors
        scores['tp'][PATHOLOGIES] = \
            (scores['y_hat'][PATHOLOGIES]).astype(float) * (scores['y'][PATHOLOGIES]).astype(float)
        scores['tn'][PATHOLOGIES] = \
            (~scores['y_hat'][PATHOLOGIES]).astype(float) * (~scores['y'][PATHOLOGIES]).astype(float)
        scores['fp'][PATHOLOGIES] = \
            (scores['y_hat'][PATHOLOGIES]).astype(float) * (~scores['y'][PATHOLOGIES]).astype(float)
        scores['fn'][PATHOLOGIES] = \
            (~scores['y_hat'][PATHOLOGIES]).astype(float) * (scores['y'][PATHOLOGIES]).astype(float)

        # Take the mean error over the DICOMs (if the sum is taken instead, studies with more DICOMs would be given more
        # importance. We want every study to be given equal importance).
        if self.accumulate_over_dicoms:
            for i in ['tp', 'tn', 'fp', 'fn']:
                scores[i] = scores[i].drop(['dicom_id'], axis=1).groupby('study_id', as_index=False).mean()

        # Initialise example scores dataframe
        scores['example'] = scores['tp'][['study_id']].copy()

        # Errors per study_id
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores['example'][f'{i}'] = scores[i][PATHOLOGIES].sum(1)

        # Scores per example
        scores['example']['accuracy'] = (
            (scores['example']['tp'] + scores['example']['tn']) /
            (scores['example']['tp'] + scores['example']['tn'] + scores['example']['fp'] + scores['example']['fn'])
        ).fillna(0)
        scores['example']['precision'] = (
            scores['example']['tp'] / (scores['example']['tp'] + scores['example']['fp'])
        ).fillna(0)
        scores['example']['recall'] = (
            scores['example']['tp'] / (scores['example']['tp'] + scores['example']['fn'])
        ).fillna(0)
        scores['example']['f1'] = (
            scores['example']['tp'] / (scores['example']['tp'] + 0.5 * (
                scores['example']['fp'] + scores['example']['fn'])
            )
        ).fillna(0)

        # Average example scores
        scores['averaged'] = pd.DataFrame(
            scores['example'].drop(['study_id', 'tp', 'tn', 'fp', 'fn'], axis=1).mean().rename('{}_example'.format)
        ).transpose()

        # Initialise class scores dataframe
        scores['class'] = pd.DataFrame()

        # Sum over study_ids for class scores
        for i in ['tp', 'tn', 'fp', 'fn']:
            scores['class'][i] = scores[i][PATHOLOGIES].sum()

        # Scores for each class
        scores['class']['accuracy'] = (
            (scores['class']['tn'] + scores['class']['tp']) / (
                scores['class']['tp'] + scores['class']['tn'] +
                scores['class']['fp'] + scores['class']['fn']
            )
        ).fillna(0)
        scores['class']['precision'] = (
            scores['class']['tp'] / (
                scores['class']['tp'] + scores['class']['fp']
            )
        ).fillna(0)
        scores['class']['recall'] = (
            scores['class']['tp'] / (
                scores['class']['tp'] + scores['class']['fn']
            )
        ).fillna(0)
        scores['class']['f1'] = (
            scores['class']['tp'] / (
                scores['class']['tp'] + 0.5 * (scores['class']['fp'] + scores['class']['fn'])
            )
        ).fillna(0)

        # Macro-averaging
        for i in ['accuracy', 'precision', 'recall', 'f1']:
            scores['averaged'][f'{i}_macro'] = [scores['class'][i].mean()]

        # Micro-averaged over the classes:
        scores['averaged']['accuracy_micro'] = (scores['class']['tp'].sum() + scores['class']['tn'].sum()) / (
            scores['class']['tp'].sum() + scores['class']['tn'].sum() +
            scores['class']['fp'].sum() + scores['class']['fn'].sum()
        )
        scores['averaged']['precision_micro'] = scores['class']['tp'].sum() / (
            scores['class']['tp'].sum() + scores['class']['fp'].sum()
        )
        scores['averaged']['recall_micro'] = scores['class']['tp'].sum() / (
            scores['class']['tp'].sum() + scores['class']['fn'].sum()
        )
        scores['averaged']['f1_micro'] = scores['class']['tp'].sum() / (
            scores['class']['tp'].sum() + 0.5 * (scores['class']['fp'].sum() + scores['class']['fn'].sum())
        )

        # Reformat classification scores for individual pathologies
        scores['class'].insert(loc=0, column='pathology', value=scores['class'].index)
        scores['class'] = scores['class'].drop(['tp', 'tn', 'fp', 'fn'], axis=1).melt(
            id_vars=['pathology'],
            var_name='metric',
            value_name='score',
        )
        scores['class']['metric'] = scores['class']['metric'] + '_' + scores['class']['pathology']
        scores['class'] = pd.DataFrame([scores['class']['score'].tolist()], columns=scores['class']['metric'].tolist())

        # Save the example and class scores
        def save_scores():
            scores['class'].to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_class_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )
            scores['example'].to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_example_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save_scores()
        elif torch.distributed.get_rank() == 0:
            save_scores()

        score_dict = {
            **scores['averaged'].to_dict(orient='records')[0],
            **scores['class'].to_dict(orient='records')[0],
            'num_study_ids': float(scores['y'].study_id.nunique()),
        }

        # Number of examples
        if self.accumulate_over_dicoms:
            score_dict['num_dicom_ids'] = float(scores['y'].dicom_id.nunique())

        return score_dict
