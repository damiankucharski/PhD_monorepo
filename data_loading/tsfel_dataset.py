from data_loading.dataset import Dataset
from functools import cached_property
from pathlib import Path
from typing import Union

import pandas as pd
from common.json import Json
import wandb

from data_loading.dataset_data import DatasetData

class TsfelDataset(Dataset):

    def __init__(self, X_train, X_val, X_test, y_train, y_val, y_test, tsfel_config, extractor_config, name = "Tsfel_dataset"):
        super(TsfelDataset, self).__init__()
        self.name = name
        self.local_path = None

        self.dataset_data = DatasetData(X_train, X_val, X_test, y_train, y_val, y_test)

        self.tsfel_config = tsfel_config
        self.extractor_config = extractor_config

    @staticmethod
    def create(path: Union[Path, str], name):
        X_train = pd.read_csv(path / "X_train.csv", index_col=0)
        X_val = pd.read_csv(path / "X_val.csv", index_col=0)
        X_test = pd.read_csv(path / "X_test.csv", index_col=0)
        y_train = pd.read_csv(path / "y_train.csv", index_col=0)
        y_val = pd.read_csv(path / "y_val.csv", index_col=0)
        y_test = pd.read_csv(path / "y_test.csv", index_col=0)
        tsfel_config = Json.load(path / 'tsfel_config.json')
        extractor_config = Json.load(path / 'extractor_config.json')

        return TsfelDataset(X_train, X_val, X_test, y_train, y_val, y_test, tsfel_config, extractor_config, name)
        

    def save_to_dir(self, path: Union[str, Path]):

        print(f"Saving to directory: {path}")

        self.local_path = Path(path)

        self.X_train.to_csv(path / "X_train.csv")
        self.X_val.to_csv(path / "X_val.csv")
        self.X_test.to_csv(path / "X_test.csv")
        self.y_train.to_csv(path / "y_train.csv")
        self.y_val.to_csv(path / "y_val.csv")
        self.y_test.to_csv(path / "y_test.csv")
        Json.save(path / 'tsfel_config.json', self.tsfel_config)
        Json.save(path / 'extractor_config.json', self.extractor_config)
        
        print("Saved")

    
    def save_wab(self, project_name, tags=['latest'], local_path=None, metadata={}, depends_on: wandb.Artifact = None):

        metadata.update({'tsfel_config':self.tsfel_config,'extractor_config':self.extractor_config})

        super().save_wab(project_name=project_name, tags=tags, local_path=local_path, metadata=metadata)
    
    @staticmethod
    def load_wab(project_name, dataset_name = "tsfel_dataset", tag='latest'):
        return Dataset.load_wab(project_name = project_name, dataset_name=dataset_name, tag=tag)

    def get_artifact_name(self, project_name, version="latest"):
        return super().get_artifact_name(project_name, version)

    @cached_property
    def X_train(self):
        return self.dataset_data.X_train

    @cached_property
    def X_val(self):
        return self.dataset_data.X_val

    @cached_property
    def X_test(self):
        return self.dataset_data.X_test

    @cached_property
    def y_train(self):
        return self.dataset_data.y_train

    @cached_property
    def y_val(self):
        return self.dataset_data.y_val

    @cached_property
    def y_test(self):
        return self.dataset_data.y_test