from pathlib import Path
from torchmetrics import Metric
import os
import pandas as pd
import time
import torch


class ReportTokenIdentifiersLogger(Metric):

    is_differentiable = False
    full_state_update = False

    """
    Logs the findings and impression section token identifiers of a report to a .csv.
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
        super().__init__(dist_sync_on_step=dist_sync_on_step, compute_on_step=False)
        self.exp_dir = exp_dir
        self.split = split
        self.track_dicom_id = track_dicom_id

        # No dist_reduce_fx, manually sync over devices
        self.add_state('reports', default=[])

        self.save_dir = os.path.join(self.exp_dir, 'generated_report_ids')

        Path(self.save_dir).mkdir(parents=True, exist_ok=True)

    def update(self, report_ids, study_ids, dicom_ids=None):
        """
        Argument/s:
            report_ids - report identifiers.
            study_ids - list of study identifiers.
            dicom_ids - list of dicom identifiers.
        """

        assert isinstance(report_ids, torch.Tensor), '"report_ids" must be a torch.Tensor.'

        if self.track_dicom_id:
            for (i_1, i_2, i_3) in zip(report_ids.tolist(), dicom_ids, study_ids):
                self.reports.append({'report_ids': i_1, 'dicom_id': i_2, 'study_id': i_3})
        else:
            for (i_1, i_2) in zip(report_ids.tolist(), study_ids):
                self.reports.append({'report_ids': i_1, 'study_id': i_2})

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
