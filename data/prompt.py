import itertools
import random
from collections import OrderedDict
import warnings

import numpy as np
import torch

from data.study_id import StudyIDSubset


class PreviousReportSubset(StudyIDSubset):
    """
    Study ID subset with the previous report given as a prompt for the decoder. Examples are indexed by the study
    identifier.
    """

    def __init__(self, history, use_generated=False, scst_generated=False, mbatch_size=None, **kwargs):
        """
        Argument/s:
            history - dataframe that provides the history for each study.
            use_generated - whether to use the generated report as the prompt.
            scst_generated - whether to train with the generated report as the prompt with SCST.
            mbatch_size - the size of the mini-batch (used when scst_generated is set).
        """
        super(PreviousReportSubset, self).__init__(**kwargs)

        self.history = history
        self.use_generated = use_generated
        self.scst_generated = scst_generated
        self.mbatch_size = mbatch_size

        """
        Subject 15964158 has two studies (57077869 and 58837588) with identical times, making it difficult to determine
        their order. Hence, these two studies, along with all following studies are dropped.
        
        21800331 is the date of studies 57077869 and 58837588.
        """
        subject_history = self.df.loc[self.df['subject_id'] == 15964158].sort_values(['StudyDate', 'StudyTime'])
        excluded_studies = subject_history[subject_history.StudyDate >= 21800331].study_id.tolist()
        self.df = self.df[~self.df.study_id.isin(excluded_studies)]

        """
        Subject 10661934 has two studies (52654465 and 52849261) with identical times, making it difficult to determine
        their order. Hence, these two studies, along with all following studies are removed.

        21490809 is the date of studies 52654465 and 52849261.
        """
        subject_history = self.df.loc[self.df['subject_id'] == 10661934].sort_values(['StudyDate', 'StudyTime'])
        excluded_studies = subject_history[subject_history.StudyDate >= 21490809].study_id.tolist()
        self.df = self.df[~self.df.study_id.isin(excluded_studies)]

        """
        Subject 16973455 has two studies (50594285 and 55190090) with identical times, making it difficult to determine
        their order. Hence, these two studies, along with all following studies are removed.

        21440406 is the date of studies 50594285 and 55190090.
        """
        subject_history = self.df.loc[self.df['subject_id'] == 16973455].sort_values(['StudyDate', 'StudyTime'])
        excluded_studies = subject_history[subject_history.StudyDate >= 21440406].study_id.tolist()
        self.df = self.df[~self.df.study_id.isin(excluded_studies)]

        # Sort by study_id, date, then time for testing with previous outputs:
        self.df = self.df.sort_values(['subject_id', 'StudyDate', 'StudyTime'], ascending=[True, True, True])

        # Need to reset self.examples as some studies have been dropped:
        self.examples = self.df['study_id'].drop_duplicates().tolist()

        # For generated previous findings & impression:
        if self.use_generated: 
            self.history['generated_findings'] = np.nan
            self.history['generated_impression'] = np.nan
            self.allocate_subjects_to_rank(shuffle_subjects=False)

        # Train with the previously generated report as the prompt with SCST:
        if self.scst_generated:

            # Initialise the examples (this needs to be called again in the LightningModule in the on_train_epoch_start hook):
            seed = 0
            self.allocate_subjects_to_rank(seed=seed)

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        # Get example dict from parent class:
        example_dict = StudyIDSubset.__getitem__(self, index)

        # Get the dataframe for the current study (the study of concern):
        example = self.df.loc[self.df['study_id'] == self.examples[index]]
        subject_id = example.iloc[0, example.columns.get_loc('subject_id')]
        study_date = example.iloc[0, example.columns.get_loc('StudyDate')]
        study_time = example.iloc[0, example.columns.get_loc('StudyTime')]

        # Get the dataframe of the subjects history:
        subject_history = self.history.loc[self.history['subject_id'] == subject_id].sort_values(['StudyDate', 'StudyTime'])

        # Remove studies after the current study date:
        subject_history = subject_history[subject_history['StudyDate'] <= study_date]

        # Remove studies after the current study time:
        subject_history = subject_history[(subject_history['StudyTime'] <= study_time) | (subject_history['StudyDate'] != study_date)]

        # List of all considered studies, ordered by time:
        considered_study_ids = list(OrderedDict.fromkeys(subject_history.study_id.to_list()))

        # Only consider the current study plus the previous studies (hence, -2):
        considered_study_ids = considered_study_ids[-2:]
        previous_study_id = considered_study_ids[0]
        current_study_id = example_dict['study_ids']

        # If the previous study has not been dropped, provide previous sections:
        example_dict['previous_findings'] = None
        example_dict['previous_impression'] = None
        if len(considered_study_ids) == 2 and (previous_study_id == self.df.study_id).any():
        
            # Ensure that the previous study_id is not the current study_id:
            assert previous_study_id != current_study_id, f'previous_study_id ({previous_study_id}) != study_id ({current_study_id})'

            # Get the previous study:
            study = subject_history.loc[subject_history.study_id == previous_study_id]
                
            if self.use_generated:
                example_dict['previous_findings']  = study.iloc[0, study.columns.get_loc('generated_findings')]
                example_dict['previous_impression'] = study.iloc[0, study.columns.get_loc('generated_impression')]

                # Assert is not NaN (if empty section, it is set as None in LightningModule):
                assert example_dict['previous_findings'] == example_dict['previous_findings'], f'previous_findings is NaN for study_id: {previous_study_id}'
                assert example_dict['previous_impression'] == example_dict['previous_impression'], f'previous_impression is NaN for study_id: {previous_study_id}'

            else:
                previous_findings = study.iloc[0, study.columns.get_loc('findings')]
                previous_impression = study.iloc[0, study.columns.get_loc('impression')]

                # If NaN, set section as None:
                example_dict['previous_findings'] = previous_findings if previous_findings == previous_findings else None
                example_dict['previous_impression'] = previous_impression if previous_impression == previous_impression else None

        return example_dict

    def allocate_subjects_to_rank(self, seed=None, shuffle_subjects=True):
        """
        Allocates subjects to different ranks.

        Argument/s:
            seed - seed for the random shuffling.
        """

        assert self.use_generated, '"use_generated" must be True.'

        if shuffle_subjects:
            assert self.scst_generated, '"scst_generated" must be True.'

        # Get info about workers:
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, 'num_workers for the dataloader must be set to 0.'
        world_size = torch.distributed.get_world_size()

        # List containing the study_ids of each subject:
        subject_study_id_lists = self.df.drop_duplicates(
            subset=['study_id'],
        ).groupby('subject_id')['study_id'].apply(list).tolist()

        # Ensure that subjects with the most studies are first:
        subject_study_id_lists.sort(key=len, reverse=True)

        # Largest lists will be first:
        self.examples = [[] for _ in range(world_size * self.mbatch_size)]
        total_lengths = [0] * world_size * self.mbatch_size
        for i in subject_study_id_lists:

            # Find the worker with the shortest number of studies:
            idx = np.argmin(total_lengths)

            # Add the studies to the worker with the least amount of studies:
            self.examples[idx].append(i)

            # Track the length:
            total_lengths[idx] += len(i)

        # If the length of self.examples is not divisible by the world size, some examples need to be oversampled.
        # This will be accounted for by the metrics as they remove repeated study_ids:
        if len([k for i in self.examples for j in i for k in j]) % (self.mbatch_size * world_size) != 0:
            
            warnings.warn('The number of examples is not divisible by the world size. ' 
                         'Adding extra studies to account for this. This needs to be accounted for outside of the dataset.')

            # Add to the rank with the fewest studies until the following is passed:
            while len([k for i in self.examples for j in i for k in j]) % (self.mbatch_size * world_size) != 0:
 
                # Find the worker with the fewest of studies:
                num_studies = []
                for idx in range(world_size):
                    num_studies.append(len([j for i in self.examples[idx] for j in i]))
                
                # Assuming that the last element of subject_study_id_lists has a length of one:
                self.examples[np.argmin(total_lengths)].append(subject_study_id_lists[-1])

        # Randomise the position of the subjects for each mini-batch element of each rank:
        if shuffle_subjects:
            random.seed(seed)
            self.examples = [list(itertools.chain(*random.sample(i, k=len(i)))) for i in self.examples]
        else:
            self.examples = [list(itertools.chain(*i)) for i in self.examples]

        # Interleave the lists for each mini-batch element of each rank so that the study_ids for a subject occur every
        # mbatch_size * world_size elements:
        self.examples = [j for i in zip(*self.examples) for j in i]

        # Ensure that the number of examples is equal to the number of unique study_ids in the set:
        assert len(set(self.examples)) == self.df.study_id.nunique() and \
            sorted(set(self.examples)) == sorted(self.df.study_id.drop_duplicates().to_list())
