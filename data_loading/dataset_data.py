from dataclasses import dataclass
import numpy as np
import pandas as pd

from typing import Union

@dataclass
class DatasetData:

    X_train: Union[pd.DataFrame, np.ndarray] = None
    X_val: Union[pd.DataFrame, np.ndarray] = None
    X_test: Union[pd.DataFrame, np.ndarray] = None
    y_train: Union[pd.DataFrame, np.ndarray] = None
    y_val: Union[pd.DataFrame, np.ndarray] = None
    y_test: Union[pd.DataFrame, np.ndarray] = None