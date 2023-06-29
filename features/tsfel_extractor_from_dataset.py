from data_loading.dataset import Dataset
from functools import cached_property
from data_processing.tsfel_features_extractor import TsfelFeaturesExtractor
import tsfel
from pathlib import Path
from typing import Union
from common.json import Json
from features.feature_extractor_from_dataset import FeatureExtractorFromDataset

class TsfelExtractorFromDataset(FeatureExtractorFromDataset):

    def __init__(self, dataset: Dataset, tsfel_config = tsfel.get_features_by_domain('temporal'), extractor_config = {'fs':100, 'window_size':1000}):
        super().__init__()
        self.parent_dataset = dataset
        self.tsfel_config = tsfel_config
        self.extractor_config = extractor_config
        self.name = "Tsfel_dataset"
        self.local_path = None
        self.features_extractor = TsfelFeaturesExtractor(self.tsfel_config, **self.extractor_config)
    
    @cached_property
    def X_train(self):

        return self.features_extractor.calculate_features(self.parent_dataset.X_train)

    @cached_property
    def X_val(self):
        return self.features_extractor.calculate_features(self.parent_dataset.X_val)
    
    @cached_property
    def X_test(self):
        return self.features_extractor.calculate_features(self.parent_dataset.X_test)


    @cached_property
    def y_train(self):
        return self.parent_dataset.y_train
    
    @cached_property
    def y_val(self):
        return self.parent_dataset.y_val

    @cached_property
    def y_test(self):
        return self.parent_dataset.y_test

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