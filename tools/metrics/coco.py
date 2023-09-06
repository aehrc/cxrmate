from pathlib import Path
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.spice.spice import Spice
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from torchmetrics import Metric
from typing import Optional
import numpy as np
import os
import pandas as pd
import re
import time
import torch


class COCONLGMetricsMIMICCXR(Metric):
    """
    COCO NLG metrics for MIMIC-CXR. Equal importance is given to each study. Thus, scores are averaged over the
    study_ids. If multiple reports are generated per study_id, its score is the mean score for each of
    its dicom_ids.
    """

    is_differentiable = False
    full_state_update = False

    def __init__(
            self,
            split: str,
            exp_dir: str,
            accumulate_over_dicoms,
            metrics: Optional[list] = None,
            use_tokenizer: bool = True,
            dist_sync_on_step: bool = False,
    ):
        """
        split - name of the dataset split.
        exp_dir - path of the experiment directory.
        accumulate_over_dicoms - accumulate scores over the DICOMs for a study.
        metrics - which metrics to use.
        use_tokenizer - use the PTBTokenizer.
        exp_dir - experiment directory to save the captions and individual scores.
        dist_sync_on_step - sync the workers at each step.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.accumulate_over_dicoms = accumulate_over_dicoms
        self.metrics = ['bleu', 'cider', 'meteor', 'rouge', 'spice'] if metrics is None else metrics
        self.metrics = [metric.lower() for metric in metrics]
        self.use_tokenizer = use_tokenizer
        self.exp_dir = exp_dir

        # No dist_reduce_fx; manually gather over devices
        self.add_state('reports', default=[])

        if 'bleu' in self.metrics:
            self.bleu = Bleu(4)
        if 'meteor' in self.metrics:
            self.meteor = Meteor()
        if 'rouge' in self.metrics:
            self.rouge = Rouge()
        if 'cider' in self.metrics:
            self.cider = Cider()
        if 'spice' in self.metrics:
            self.spice = Spice()
        if self.use_tokenizer:
            self.tokenizer = PTBTokenizer()

        self.split = split

        self.save_dir = os.path.join(self.exp_dir, 'nlg_scores')
        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

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
        """
        Compute the metrics from the COCO captioning task with and without DDP.

        Argument/s:
            epoch - the training epoch.

        Returns:
            Dictionary containing the scores for each of the metrics
        """

        # Manually gather as automatically gathering strings is not supported by torchmetrics
        if torch.distributed.is_initialized():  # If DDP
            reports_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(reports_gathered, self.reports)
            self.reports = [j for i in reports_gathered for j in i]

        predictions, labels = {}, {}
        key = 'dicom_id' if self.accumulate_over_dicoms else 'study_id'
        if self.use_tokenizer:
            for i in self.reports:
                idx = i[key].item() if isinstance(i[key], torch.Tensor) else i[key]
                idx = int(idx) if isinstance(idx, np.int64) else idx  # SPICE cannot handle numpy.int.
                predictions[idx] = [{'caption': re.sub(' +', ' ', i['prediction'])}]
                labels[idx] = [{'caption': re.sub(' +', ' ', m)} for m in i['label']]

            predictions = self.tokenizer.tokenize(predictions)
            labels = self.tokenizer.tokenize(labels)

        else:
            for i in self.reports:
                idx = i[key].item() if isinstance(i[key], torch.Tensor) else i[key]
                idx = int(idx) if isinstance(idx, np.int64) else idx  # SPICE cannot handle numpy.int.
                predictions[idx] = [re.sub(' +', ' ', i['prediction'])]
                labels[idx] = [re.sub(' +', ' ', m) for m in i['label']]

        # Assume that the order of the scores is the same as the order of the dicom_ids.
        df = pd.DataFrame()
        if self.accumulate_over_dicoms:
            df['dicom_id'] = [i['dicom_id'] for i in self.reports]
        df['study_id'] = [i['study_id'] for i in self.reports]
        df = df.drop_duplicates(subset=[key])

        # Number of examples:
        scores = {'num_study_ids': float(df.study_id.nunique())}
        if self.accumulate_over_dicoms:
            scores['num_dicom_ids'] = float(df.dicom_id.nunique())

        # COCO NLG metric scores:
        if 'bleu' in self.metrics:
            _, metric_scores = self.bleu.compute_score(labels, predictions)
            df['bleu_1'] = metric_scores[0]
            df['bleu_2'] = metric_scores[1]
            df['bleu_3'] = metric_scores[2]
            df['bleu_4'] = metric_scores[3]
        if 'meteor' in self.metrics:
            _, metric_scores = self.meteor.compute_score(labels, predictions)
            df['meteor'] = metric_scores
        if 'rouge' in self.metrics:
            _, metric_scores = self.rouge.compute_score(labels, predictions)
            df['rouge'] = metric_scores
        if 'cider' in self.metrics:
            _, metric_scores = self.cider.compute_score(labels, predictions)
            df['cider'] = metric_scores
        if 'spice' in self.metrics:
            _, metric_scores = self.spice.compute_score(labels, predictions)
            df['spice'] = metric_scores

        def save():
            df.to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()

        # Mean and max over the scores for each DICOM:
        if self.accumulate_over_dicoms:
            df = df.drop(['dicom_id'], axis=1).groupby('study_id', as_index=False).mean()

        # Mean scores over the study identifiers
        df = df.drop(['study_id'], axis=1).mean()
        scores = {**scores, **df.to_dict()}

        return scores
