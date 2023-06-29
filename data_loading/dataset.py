from abc import ABC, abstractmethod
from functools import cached_property
import wandb
from typing import Union
from pathlib import Path

class Dataset(ABC):

    def __init__(self, name = ""):
        self.name = name
        self.local_path = None

    @staticmethod
    @abstractmethod
    def create():
        pass

    @abstractmethod
    def save_to_dir(self, path: Union[str, Path]):
        pass

    @abstractmethod
    def save_wab(self, project_name, tags=['latest'], local_path=None, metadata={}):
        if (local_path is None) and (self.local_path is None):
            raise Exception("Use save_to_dir() first or provide local_path argument")

        if local_path is not None:
            self.local_path = local_path

        self.wab_save_run = wandb.init(
            project=project_name, 
            job_type='upload-dataset'
            )

        self.output_artifact = wandb.Artifact(
        name=self.name, 
        type='dataset',
        metadata=metadata
        )  

        self.output_artifact.add_dir(str(self.local_path))

        self.wab_save_run.log_artifact(self.output_artifact, aliases=tags)

        self.wab_save_run.finish()

        return self.output_artifact
    
    @staticmethod
    @abstractmethod
    def load_wab(project_name, tag='latest'):
        pass

    @abstractmethod
    @cached_property 
    def X_train(self):
        pass


    @abstractmethod
    @cached_property 
    def X_val(self):
        pass


    @abstractmethod
    @cached_property 
    def X_test(self):
        pass

    @abstractmethod
    @cached_property 
    def y_train(self):
        pass

    @abstractmethod
    @cached_property 
    def y_val(self):
        pass

    @abstractmethod
    @cached_property 
    def y_test(self):
        pass