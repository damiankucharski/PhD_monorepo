from pathlib import Path

from data_loading.ptb_xl_3d_dataset import PTB_XL_3D_dataset
from features.tsfel_extractor_from_dataset import TsfelExtractorFromDataset
from data_processing.tsfel_features_extractor import TsfelFeaturesExtractor
import tsfel

if __name__ == "__main__":

    local_path = Path(__file__).parent / '../data/ptb_xl_dataset_tsfel_features'
    local_path.mkdir(parents=True, exist_ok=True)

    print("loading ptb xl dataset from wab")

    dataset = PTB_XL_3D_dataset.load_wab()

    print(dataset.X_test.shape)

    print("Creating extractor")

    extractor = TsfelExtractorFromDataset(dataset)
    extractor.save_to_dir(local_path)
    extractor.save_wab('ecg', local_path=local_path)
