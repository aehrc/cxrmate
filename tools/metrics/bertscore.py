import os
import time
from pathlib import Path

import pandas as pd
import torch
from bert_score import BERTScorer
from torchmetrics import Metric

# from torchmetrics.text import BERTScore
from transformers import AutoModel, AutoTokenizer


class BERTScoreRoBERTaLarge(Metric):
    """
    BERTScore for MIMIC-CXR. If multiple reports are generated per study_id, each error type is
    summed over the dicom_ids.
    """

    def __init__(
            self, split, ckpt_dir, mbatch_size, exp_dir, accumulate_over_dicoms, num_workers,
    ):
        """
        Argument/s:
            split - dataset split.
            ckpt_dir - path to the checkpoint directory.
            mbatch_size - mini-batch size for CheXbert.
            exp_dir - experiment directory where outputs will be saved.
            accumulate_over_dicoms - whether to accumulate scores over the report for each DICOM for a study.
            num_workers - the number of workers for BERTScore.
        """
        super().__init__(dist_sync_on_step=False)

        self.split = split
        self.ckpt_dir = ckpt_dir
        self.mbatch_size = mbatch_size
        self.exp_dir = exp_dir
        self.accumulate_over_dicoms = accumulate_over_dicoms
        self.num_workers = num_workers

        self.add_state('reports', default=[])

        self.save_dir = os.path.join(self.exp_dir, 'bertscore')
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

        # BertScore:
        bert_scorer = BERTScorer(
            model_type='roberta-large',
            num_layers=17,
            batch_size=self.mbatch_size,
            nthreads=self.num_workers,
            all_layers=False,
            idf=False,
            lang='en',
            device=self.device,
            rescale_with_baseline=True,
            # baseline_path=os.path.join(self.ckpt_dir, 'bert_score', 'rescale_baseline', 'en', 'roberta-large.tsv'),
        )

        # RoBERTa tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(os.path.join(self.ckpt_dir, 'roberta-large'))
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'

        y_hat = [j['prediction'] for j in self.reports]
        y = [j['label'] for j in self.reports]
        study_ids = [j['study_id'] for j in self.reports]
        if self.accumulate_over_dicoms:
            dicom_ids = [j['dicom_id'] for j in self.reports]

        # Following COCO, the labels are contained in a nested list:
        for j in y:
            assert len(j) == 1
        y = [j[0] for j in y]

        # To fix the issue caused by DistilBERT adding too many special tokens:
        y_hat_trimmed = tokenizer.batch_decode(
            [i[:511] for i in tokenizer(y_hat).input_ids], skip_special_tokens=True,
        )
        y_trimmed = tokenizer.batch_decode(
            [i[:511] for i in tokenizer(y).input_ids], skip_special_tokens=True,
        )

        # precision = [0.0] * len(y_trimmed)
        # recall = [0.0] * len(y_trimmed)
        # f1 = [0.0] * len(y_trimmed)
        #
        # # Drop pairs and track indices:
        # y_hat_checked, y_checked, indices = [], [], []
        # for i, (j, k) in enumerate(zip(y_hat_trimmed, y_trimmed)):
        #     if j and k:
        #         y_hat_checked.append(j)
        #         y_checked.append(k)
        #         indices.append(i)
        #     elif not y_hat_trimmed and y_trimmed:
        #         precision[i] = -1.0
        #         recall[i] = -1.0
        #         f1[i] = -1.0
        #
        # with torch.no_grad():
        #     bert_scores, hash_code = bert_scorer.score(y_hat_checked, y_checked, batch_size=self.mbatch_size, return_hash=True)
        # print(hash_code)
        #
        # for i, x, y, z in zip(indices, bert_scores[0].tolist(), bert_scores[1].tolist(), bert_scores[2].tolist()):
        #     precision[i] = x
        #     recall[i] = y
        #     f1[i] = z

        with torch.no_grad():
            bert_scores, hash_code = bert_scorer.score(y_hat_trimmed, y_trimmed, batch_size=self.mbatch_size, return_hash=True)
        print(hash_code)

        precision = bert_scores[0].tolist()
        recall = bert_scores[1].tolist()
        f1 = bert_scores[2].tolist()

        rows = []
        if self.accumulate_over_dicoms:
            for x, y, s_1, s_2, s_3 in zip(dicom_ids, study_ids, f1, precision, recall):
                rows.append({'dicom_id': x, 'study_id': y, 'f1': s_1, 'precision': s_2, 'recall': s_3})
        else:
            for x, s_1, s_2, s_3 in zip(study_ids, f1, precision, recall):
                rows.append({'study_id': x, 'f1': s_1, 'precision': s_2, 'recall': s_3})

        # Gather if DDP
        if torch.distributed.is_initialized():
            rows_gathered = [None] * torch.distributed.get_world_size()

            torch.distributed.all_gather_object(rows_gathered, rows)

            rows = [j for i in rows_gathered for j in i]

        bert_scores = pd.DataFrame(rows)

        # Drop duplicates caused by DDP
        key = 'dicom_id' if self.accumulate_over_dicoms else 'study_id'
        bert_scores = bert_scores.drop_duplicates(subset=[key])

        # Save the example and class scores
        def save_scores():
            bert_scores.to_csv(
                os.path.join(
                    self.save_dir,
                    f'{self.split}_epoch-{epoch}_scores_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv',
                ),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save_scores()
        elif torch.distributed.get_rank() == 0:
            save_scores()

        # Take the mean error over the DICOMs (if the sum is taken instead, studies with more DICOMs would be given more
        # importance. We want every study to be given equal importance).
        if self.accumulate_over_dicoms:
            bert_scores = bert_scores.drop(['dicom_id'], axis=1).groupby('study_id', as_index=False).mean()

        return {
            'f1': bert_scores.f1.mean().item(),
            'precision': bert_scores.precision.mean().item(),
            'recall': bert_scores.recall.mean().item(),
        }
