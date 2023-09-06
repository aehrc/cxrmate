from pathlib import Path
from torchmetrics import Metric
import os
import pandas as pd
import time
import torch


class ReportLogger(Metric):

    is_differentiable = False
    full_state_update = False

    """
    Logs the findings and impression sections of a report to a .csv.
    """

    def __init__(
            self,
            exp_dir: str,
            split: str,
            track_dicom_id: bool,
            dist_sync_on_step: bool = False,
    ):
        """
        exp_dir - experiment directory to save the captions and individual scores.
        split - train, val, or test split.
        track_dicom_id - track the DICOM identifier if generating a report per DICOM.
        dist_sync_on_step - sync the workers at each step.
        """
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.exp_dir = exp_dir
        self.split = split
        self.track_dicom_id = track_dicom_id

        # No dist_reduce_fx, manually sync over devices
        self.add_state('reports', default=[])

        self.save_dir = os.path.join(self.exp_dir, 'generated_reports')

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def update(self, findings, impression, study_ids, dicom_ids=None):
        """
        Argument/s:
            findings - the findings section must be in the following format:

                [
                    '...',
                    '...',
                ]
            impression - the impression section must be in the following format:

                [
                    '...',
                    '...',
                ]
            study_ids - list of study identifiers.
            dicom_ids - list of dicom identifiers.
        """

        assert isinstance(findings, list), '"findings" must be a list of strings.'
        assert all(isinstance(i, str) for i in findings), 'Each element of "findings" must be a string.'
        assert isinstance(impression, list), '"impression" must be a list of strings.'
        assert all(isinstance(i, str) for i in impression), 'Each element of "impression" must be a string.'

        if self.track_dicom_id:
            for (i_1, i_2, i_3, i_4) in zip(findings, impression, dicom_ids, study_ids):
                self.reports.append({'findings': i_1, 'impression': i_2, 'dicom_id': i_3, 'study_id': i_4})
        else:
            for (i_1, i_2, i_3) in zip(findings, impression, study_ids):
                self.reports.append({'findings': i_1, 'impression': i_2, 'study_id': i_3})

    def compute(self, epoch):
        if torch.distributed.is_initialized():  # If DDP
            reports_gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(reports_gathered, self.reports)
            self.reports = [j for i in reports_gathered for j in i]

        return self.log(epoch)

    def log(self, epoch):

        def save():

            key = 'dicom_id' if self.track_dicom_id else 'study_id'
            df = pd.DataFrame(self.reports).drop_duplicates(subset=key)

            df.to_csv(
                os.path.join(self.save_dir, f'{self.split}_epoch-{epoch}_{time.strftime("%d-%m-%Y_%H-%M-%S")}.csv'),
                index=False,
            )

        if not torch.distributed.is_initialized():
            save()
        elif torch.distributed.get_rank() == 0:
            save()
