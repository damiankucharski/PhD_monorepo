from data_loading.dataset import Dataset
import tsfel

import pandas as pd
import numpy as np
import wfdb
import ast
from pathlib import Path
import os

class PTB_XL_Dataset(Dataset):

    def __init__(self, dataset_dict):
        self.dataset_dict = dataset_dict

    @staticmethod
    def create(path_to_data, sampling_rate):

        print("loading")

        Y = pd.read_csv(path_to_data / 'ptbxl_database.csv', index_col='ecg_id')
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
        Y.patient_id = Y.patient_id.astype(int)
        Y.nurse = Y.nurse.astype('Int64')
        Y.site = Y.site.astype('Int64')
        Y.validated_by = Y.validated_by.astype('Int64')

        # Load raw signal data
        X = PTB_XL_Dataset._load_raw_data(Y, sampling_rate, path_to_data)

        # Load scp_statements.csv for diagnostic aggregation
        agg_df = pd.read_csv(path_to_data / 'scp_statements.csv', index_col=0)
        agg_df = agg_df[agg_df.diagnostic == 1]

        def diagnostic_class(scp):
            res = set()
            for k in scp.keys():
                if k in agg_df.index:
                    res.add(agg_df.loc[k].diagnostic_class)
            return list(res)

        Y['scp_classes'] = Y.scp_codes.apply(diagnostic_class)

        Z = pd.DataFrame(0, index=Y.index, columns=['NORM', 'MI', 'STTC', 'CD', 'HYP'], dtype='int')
        for i in Z.index:
            for k in Y.loc[i].scp_classes:
                Z.loc[i, k] = 1

        print("Splitting")

        # Split data into train and test
        test_fold = 10
        val_fold = 9
        # Train
        X_train = X[np.where(Y.strat_fold != test_fold)]
        y_train = Z[(Y.strat_fold != test_fold)]
        # Test
        X_test = X[np.where(Y.strat_fold == test_fold)]
        y_test = Z[Y.strat_fold == test_fold]
        # Val
        X_val = X[np.where(Y.strat_fold == val_fold)]
        y_val = Z[Y.strat_fold == val_fold]

        # Extract features
        cfg = tsfel.get_features_by_domain('temporal')

        print("Extracting univariate_tsfel")

        # Train, Test, Val data for univariate_tsfel
        X_train_uni = np.array([tsfel.time_series_features_extractor(cfg, x, fs=100).values for x in X_train])
        X_test_uni = np.array([tsfel.time_series_features_extractor(cfg, x, fs=100).values for x in X_test])
        X_val_uni = np.array([tsfel.time_series_features_extractor(cfg, x, fs=100).values for x in X_val])

        print("Extracting multivatiate")

        # Train, Test, Val data for multivariate_tsfel
        X_train_multi = np.array([tsfel.time_series_features_extractor(cfg, x.reshape(-1, 1), fs=100).values for x in X_train])
        X_test_multi = np.array([tsfel.time_series_features_extractor(cfg, x.reshape(-1, 1), fs=100).values for x in X_test])
        X_val_multi = np.array([tsfel.time_series_features_extractor(cfg, x.reshape(-1, 1), fs=100).values for x in X_val])

        dataset_dict = {
            'original': {
                'X_train': X_train,
                'X_val': X_val,
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            },
            'univariate_tsfel': {
                'X_train': X_train_uni,
                'X_val': X_val_uni,
                'X_test': X_test_uni,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            },
            'multivariate_tsfel': {
                'X_train': X_train_multi,
                'X_val': X_val_multi,
                'X_test': X_test_multi,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
        }

        return PTB_XL_Dataset(dataset_dict)

    @staticmethod
    def _load_raw_data(df, sampling_rate, path):
        if sampling_rate == 100:
            data = [wfdb.rdsamp(path / f) for f in df.filename_lr]
        else:
            data = [wfdb.rdsamp(path / f) for f in df.filename_hr]
        data = np.array([signal for signal, meta in data])
        return data