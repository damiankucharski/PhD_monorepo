from data_loading.dataset import Dataset
from functools import cached_property

import pandas as pd
import numpy as np
from pathlib import Path

import wandb

from typing import Union

class PTB_XL_Dataset(Dataset):

    def __init__(self, features, metadata, name="ptb_xl_dataset"):
        super().__init__(name=name)
        self.features = features
        self.metadata = metadata
        self.local_path = None
        self.name = name

    @staticmethod
    def create(path_to_data = Path("../../Data/ptb_xl_dataset_reformatted"), name = "ptb_xl_dataset"):

        print("Reading meta")

        test_meta = X_test = pd.read_csv(path_to_data/ "test_meta.csv")
        val_meta = X_test = pd.read_csv(path_to_data/ "valid_meta.csv")
        train_meta = X_test = pd.read_csv(path_to_data/ "train_meta.csv")

        meta = pd.concat([test_meta, val_meta, train_meta]).sort_values('ecg_id')
        meta.index = meta.ecg_id
        meta.index.name = 'ecg_id'

        print("Reading features")

        X_train = pd.read_csv(path_to_data/ "train_signal.csv")
        X_val = pd.read_csv(path_to_data/ "valid_signal.csv")
        X_test = pd.read_csv(path_to_data/ "test_signal.csv")

        print("Concatenating features")

        features  = pd.concat([X_train, X_val, X_test]).reset_index().sort_values(['ecg_id','index'])
        features.index = features.index

        print("Done")

        return PTB_XL_Dataset(features, meta, name)

    
    def save_to_dir(self, path: Union[str, Path]):

        self.local_path = Path(path)

        self.features.to_csv(self.local_path / 'ptb_xl_features.csv')
        self.metadata.to_csv(self.local_path / 'ptb_xl_meta.csv')


    def save_wab(self, project_name='ecg', tags=['latest'], local_path=None, metadata={}):

        return super().save_wab(self, project_name=project_name, tags=tags, local_path=local_path, metadata=metadata)
        
    @staticmethod
    def load_wab(project_name='ecg', dataset_name = 'ptb_xl_dataset', tag='latest'):
        
        run = wandb.init(
        project=project_name, 
        job_type='download-dataset'
        )

        artifact = run.use_artifact(f'{project_name}/{dataset_name}:{tag}')
        download_path = Path(artifact.download())

        features = pd.read_csv(download_path/'ptb_xl_features.csv', index_col=0)
        meta = pd.read_csv(download_path/'ptb_xl_meta.csv', index_col=0)

        run.finish()

        return PTB_XL_Dataset(features, meta)
    
    @cached_property
    def X_train(self):
        return self.features[self.features.ecg_id.isin(self.metadata[self.metadata['strat_fold']<=8].index)]

    @cached_property
    def X_val(self):
        return self.features[self.features.ecg_id.isin(self.metadata[self.metadata['strat_fold']==9].index)]
    
    @cached_property
    def X_test(self):
        return self.features[self.features.ecg_id.isin(self.metadata[self.metadata['strat_fold']==10].index)]
    
    @cached_property
    def y_train(self):
        return self.metadata.loc[self.metadata['strat_fold'] <= 8, ['NORM','MI','STTC','HYP','CD']]
    
    @cached_property
    def y_val(self):
        return self.metadata.loc[self.metadata['strat_fold'] == 9, ['NORM','MI','STTC','HYP','CD']]
    
    @cached_property
    def y_test(self):
        return self.metadata.loc[self.metadata['strat_fold'] == 10, ['NORM','MI','STTC','HYP','CD']]
    
    @cached_property
    def X_train_reshaped(self):
        return self.X_train.values.reshape(-1, 1000, self.X_train.shape[-1])[..., -12:]

    @cached_property
    def X_val_reshaped(self):
        return self.X_val.values.reshape(-1, 1000, self.X_val.shape[-1])[..., -12:]

    @cached_property
    def X_test_reshaped(self):
        return self.X_test.values.reshape(-1, 1000, self.X_test.shape[-1])[..., -12:]
    
    def get_artifact_name(self, project_name, version="latest"):
        return super().get_artifact_name(project_name, version)