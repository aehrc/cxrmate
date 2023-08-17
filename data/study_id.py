import torch

from data.dicom_id import DICOMIDSubset


class StudyIDSubset(DICOMIDSubset):
    """
    Study ID subset. Examples are indexed by the study identifier.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # For multi-image, the study identifiers make up the training examples:
        self.examples = self.df['study_id'].drop_duplicates().tolist()

        # Column:
        self.column = 'study_id'

    def get_images(self, example):
        """
        Get the image/s for a given example. 

        Argument/s:
            example - dataframe for the example.

        Returns:
            The image/s for the example
        """
        # Load and pre-process each CXR:
        images = []
        for _, row in example.iterrows():
            images.append(self.load_and_preprocess_image(row['subject_id'], row['study_id'], row['dicom_id']))
        return torch.stack(images, 0)

    def __getitem__(self, index):

        # Get example dict from parent class:
        example_dict = DICOMIDSubset.__getitem__(self, index)

        return example_dict