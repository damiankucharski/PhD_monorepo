from data_loading.ptb_xl_dataset import PTB_XL_Dataset
from functools import cached_property
from pathlib import Path


class PTB_XL_3D_dataset(PTB_XL_Dataset):


    def __init__(self, features, metadata, name="PTB_XL_3D_Dataset"):
        super().__init__(features, metadata, name=name)

        temp_X_test = super().X_test_reshaped
        self.X_test = temp_X_test
        self.X_test_reshaped = temp_X_test

        temp_X_val = super().X_val_reshaped
        self.X_val = temp_X_val
        self.X_val_reshaped = temp_X_val
        
        temp_X_train = super().X_train_reshaped
        self.X_train = temp_X_train
        self.X_train_reshaped = temp_X_train

    @staticmethod
    def create(path_to_data = Path("../../Data/ptb_xl_dataset_reformatted")):
        parent_dataset = PTB_XL_Dataset.create(path_to_data)
        this_dataset = PTB_XL_3D_dataset(parent_dataset.features, parent_dataset.metadata)
        return this_dataset
    
    @staticmethod
    def load_wab(project_name='ecg', tag='latest'):
        parent_dataset =  PTB_XL_Dataset.load_wab(project_name, tag)
        this_dataset = PTB_XL_3D_dataset(parent_dataset.features, parent_dataset.metadata)
        return this_dataset
