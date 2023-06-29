from pathlib import Path
from data_loading.ptb_xl_dataset import PTB_XL_Dataset
import os

local_path = Path(__file__).parent / '../data/ptb_xl_dataset'

dataset_path = Path(__file__).parent / '../../Data/ptb_xl_dataset_reformatted'
 

dataset = PTB_XL_Dataset.create(dataset_path)
print("saving locally")


print(os.listdir(dataset_path))

dataset.save_to_dir(local_path)

print("Saving to wab")

dataset.save_wab()

print("Weighting for upload to finish and finishing run")