from data_loading.ptb_xl_dataset import PTB_XL_Dataset
from functools import cached_property
from pathlib import Path


class PTB_XL_3D_dataset(PTB_XL_Dataset):


    def __init__(self, features, metadata, name="ptb_xl_dataset"):
        super(PTB_XL_3D_dataset, self).__init__(features=features, metadata=metadata, name=name)

    @staticmethod
    def create(path_to_data = Path("../../Data/ptb_xl_dataset_reformatted"), name="ptb_xl_dataset"):
        parent_dataset = PTB_XL_Dataset.create(path_to_data,name)
        this_dataset = PTB_XL_3D_dataset(parent_dataset.features, parent_dataset.metadata)
        return this_dataset
    
    @staticmethod
    def load_wab(project_name='ecg', dataset_name = "ptb_xl_dataset", tag='latest'):
        parent_dataset =  PTB_XL_Dataset.load_wab(project_name, tag)
        this_dataset = PTB_XL_3D_dataset(parent_dataset.features, parent_dataset.metadata)
        return this_dataset

    @cached_property
    def X_train(self):
        features_not_reshaped = self.features[self.features.ecg_id.isin(self.metadata[self.metadata['strat_fold'] <= 8].index)]
        return features_not_reshaped.values.reshape(-1, 1000, features_not_reshaped.shape[-1])[..., -12:]

    @cached_property
    def X_val(self):
        features_not_reshaped = self.features[self.features.ecg_id.isin(self.metadata[self.metadata['strat_fold'] == 9].index)]
        return features_not_reshaped.values.reshape(-1, 1000, features_not_reshaped.shape[-1])[..., -12:]

    @cached_property
    def X_test(self):
        features_not_reshaped = self.features[self.features.ecg_id.isin(self.metadata[self.metadata['strat_fold'] == 10].index)]
        return features_not_reshaped.values.reshape(-1, 1000, features_not_reshaped.shape[-1])[..., -12:]

    @cached_property
    def X_train_reshaped(self):
        raise ValueError("This dataset already contains reshaped features")

    @cached_property
    def X_val_reshaped(self):
        raise ValueError("This dataset already contains reshaped features")

    @cached_property
    def X_test_reshaped(self):
        raise ValueError("This dataset already contains reshaped features")