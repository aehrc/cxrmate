from PIL import Image
from torch.utils.data import Dataset

from tools.utils import mimic_cxr_image_path


class DICOMIDSubset(Dataset):
    """
    DICOM ID subset. Examples are indexed by the DICOM identifier.
    """
    def __init__(self, df, dataset_dir, transforms, colour_space='RGB'):
        """
        Argument/s:
            df - Dataframe containing the splits of MIMIC-CXR.
            dataset_dir - Dataset directory.
            transforms - torchvision transformations.
            colour_space - PIL target colour space.
        """
        super(DICOMIDSubset, self).__init__()
        self.df = df
        self.dataset_dir = dataset_dir
        self.transforms = transforms
        self.colour_space = colour_space

        # The DICOM identifiers make up the training examples:
        self.examples = self.df['dicom_id'].drop_duplicates().tolist()

        # Column:
        self.column = 'dicom_id'

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):

        # Get the dataframe for the DICOM:
        example = self.df.loc[self.df[self.column] == self.examples[index]]
        study_id = example.iloc[0, example.columns.get_loc('study_id')]
        dicom_id = example.iloc[0, example.columns.get_loc('dicom_id')]

        # Get image/s:
        images = self.get_images(example)

        # Findings section:
        findings = example.iloc[0, example.columns.get_loc('findings')]
        findings = findings if findings == findings else None  # Set as None if NaN.

        # Impression section:
        impression = example.iloc[0, example.columns.get_loc('impression')]
        impression = impression if impression == impression else None  # Set as None if NaN.

        return {
            'images': images, 
            'findings': findings,
            'impression': impression,
            'dicom_ids': dicom_id, 
            'study_ids': study_id,
        }
    
    def get_images(self, example):
        """
        Get the image/s for a given example. For DICOMIDSubset, this will only be
        one image.

        Argument/s:
            example - dataframe for the example.

        Returns:
            The image/s for the example
        """
        subject_id = example.iloc[0, example.columns.get_loc('subject_id')]
        study_id = example.iloc[0, example.columns.get_loc('study_id')]
        dicom_id = example.iloc[0, example.columns.get_loc('dicom_id')]

        # Load and pre-process the CXR:
        return self.load_and_preprocess_image(subject_id, study_id, dicom_id)

    def load_and_preprocess_image(self, subject_id, study_id, dicom_id):
        """
        Load and preprocess an image.

        Argument/s:
             subject_id - subject identifier.
             study_id - study identifier.
             dicom_id - DICOM identifier.

        Returns:
            image - Tensor of the CXR.
        """
        image_file_path = mimic_cxr_image_path(self.dataset_dir, subject_id, study_id, dicom_id, 'jpg')
        image = Image.open(image_file_path)
        image = image.convert(self.colour_space)  # 'L' (greyscale) or 'RGB'.
        if self.transforms is not None:
            image = self.transforms(image)
        return image
