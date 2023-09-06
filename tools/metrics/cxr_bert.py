from pathlib import Path
from torchmetrics import Metric
from transformers import AutoModel, AutoTokenizer
import os
import pandas as pd
import time
import torch


class CXRBERT(Metric):
    """
    CXR-BERT similarity for MIMIC-CXR. If multiple reports are generated per study_id, each error type is
    summed over the dicom_ids.
    """

    def __init__(
            self, split, ckpt_dir, mbatch_size, exp_dir, accumulate_over_dicoms,
    ):
        """
        Argument/s:
            split - dataset split.
            ckpt_dir - path to the checkpoint directory.
            mbatch_size - mini-batch size for CheXbert.
            exp_dir - experiment directory where outputs will be saved.
            accumulate_over_dicoms - whether to accumulate scores over the report for each DICOM for a study.
        """
        super().__init__(dist_sync_on_step=False)

        self.split = split
        self.ckpt_dir = ckpt_dir
        self.mbatch_size = mbatch_size
        self.exp_dir = exp_dir
        self.accumulate_over_dicoms = accumulate_over_dicoms

        self.add_state('reports', default=[])

        self.save_dir = os.path.join(self.exp_dir, 'cxr_bert')
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

        # Load the model and tokenizer
        ckpt_name = 'microsoft/BiomedVLP-CXR-BERT-specialized'
        tokenizer = AutoTokenizer.from_pretrained(ckpt_name, cache_dir=self.ckpt_dir, trust_remote_code=True)
        model = AutoModel.from_pretrained(ckpt_name, cache_dir=self.ckpt_dir, trust_remote_code=True).to(self.device)
        model.eval()

        rows = []
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

            with torch.no_grad():

                # Tokenize and compute the sentence embeddings
                tokenizer_output = tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=y_hat,
                    add_special_tokens=True,
                    padding='longest',
                    return_tensors='pt',
                    truncation=True,
                    max_length=model.config.max_position_embeddings,
                )

                prediction_embeddings = model(
                    input_ids=tokenizer_output.input_ids.to(self.device),
                    attention_mask=tokenizer_output.attention_mask.to(self.device),
                    output_cls_projected_embedding=True,
                    return_dict=True,
                )

                tokenizer_output = tokenizer.batch_encode_plus(
                    batch_text_or_text_pairs=y,
                    add_special_tokens=True,
                    padding='longest',
                    return_tensors='pt',
                    truncation=True,
                    max_length=model.config.max_position_embeddings,
                )

                label_embeddings = model(
                    input_ids=tokenizer_output.input_ids.to(self.device),
                    attention_mask=tokenizer_output.attention_mask.to(self.device),
                    output_cls_projected_embedding=True,
                    return_dict=True,
                )

                # Compute the cosine similarity of sentence embeddings obtained from input text prompts.
                sim = torch.nn.functional.cosine_similarity(
                    prediction_embeddings.cls_projected_embedding,
                    label_embeddings.cls_projected_embedding,
                )

            mbatch_rows = []
            if self.accumulate_over_dicoms:
                for x, y, z in zip(dicom_ids, study_ids, sim.tolist()):
                    mbatch_rows.append({'dicom_id': x, 'study_id': y, 'similarity': z})
            else:
                for x, y in zip(study_ids, sim.tolist()):
                    mbatch_rows.append({'study_id': x, 'similarity': y})

            rows.extend(mbatch_rows)

        # Gather if DDP
        if torch.distributed.is_initialized():
            rows_gathered = [None] * torch.distributed.get_world_size()

            torch.distributed.all_gather_object(rows_gathered, rows)

            rows = [j for i in rows_gathered for j in i]

        cxrbert = pd.DataFrame(rows)

        # Drop duplicates caused by DDP
        key = 'dicom_id' if self.accumulate_over_dicoms else 'study_id'
        cxrbert = cxrbert.drop_duplicates(subset=[key])

        # Save the example and class scores
        def save_scores():
            cxrbert.to_csv(
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
            cxrbert = cxrbert.drop(['dicom_id'], axis=1).groupby('study_id', as_index=False).mean()

        return cxrbert.similarity.mean()
