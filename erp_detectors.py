
import os
import sys
from contextlib import nullcontext

import torch
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

from data.erp_fraud.erpDataset import ERPDataset
from anomaly_detection.util import evaluate_detector, get_score_dict, TorchRandomSeed
from anomaly_detection.autoencoder_torch import Autoencoder
from anomaly_detection.pyod_wrapper import PyodDetector


def detect_anomalies(algorithm,
                     train_path,
                     test_path,
                     eval_path,
                     info_path,
                     split_id,
                     experiment_name,
                     categorical='onehot',
                     numeric='log10',
                     numeric_nan_bucket=False,
                     params=None,
                     output_scores=False,
                     seed=0):
    dataset = ERPDataset(train_path=train_path,
                         test_path=test_path,
                         eval_path=eval_path,
                         info_path=info_path,
                         split_id=split_id,
                         numeric_preprocessing=numeric,
                         categorical_preprocessing=categorical,
                         numeric_nan_bucket=numeric_nan_bucket,
                         seed=seed)

    X_train, y_train, X_eval, y_eval, X_test, y_test, num_prep, cat_prep = dataset.preprocessed_data.values()

    if algorithm == 'IsolationForest':
        detector_class = IsolationForest
        params['random_state'] = seed
    elif algorithm == 'OneClassSVM':
        detector_class = OneClassSVM
    elif algorithm == 'LocalOutlierFactor':
        detector_class = LocalOutlierFactor
    elif algorithm == 'Autoencoder':
        detector_class = Autoencoder
        if 'rashomon' in params['save_path']:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except NameError or ModuleNotFoundError:
                pass
            params['save_path'] = params['save_path'].format(str(seed))
        params['n_inputs'] = X_train.shape[1]
        params['seed'] = seed  # model seed
    elif algorithm in ['LOCI', 'ABOD', 'PCA', 'pyod_AE']:
        detector_class = PyodDetector
        params['algorithm'] = algorithm  # special case, need to add to hyperparams when changing to class
    else:
        raise ValueError(f"Variable algorithm was: {algorithm}")

    # path to save autoencoder model
    save_path = None
    if 'save_path' in params.keys():
        save_path = params.pop('save_path')

    # Training
    if algorithm == 'Autoencoder':
        with TorchRandomSeed(params['seed']) if 'rashomon' in params['save_path'] else nullcontext:  # rashomon seed
            detector = detector_class(**params)
        with TorchRandomSeed(42) if 'rashomon' in params['save_path'] else nullcontext:  # rashomon data loader seed
            detector = detector.fit(X_train, device=params.get('device', 'cpu'))
    else:
        detector = detector_class(**params).fit(X_train)

    # Anomaly classification outputs
    out_dict = params.copy()
    out_scores = {}
    for split in ['eval', 'test']:
        path = eval_path if split == 'eval' else test_path
        X_task = X_eval if split == 'eval' else X_test
        if path is None:
            continue

        if 'erp_fraud' in path:
            y_task = dataset.make_labels_fraud_only(split)
        elif 'erp_mas' in path:
            y_task = dataset.preprocessed_data[f"y_{split}"]
            y_task = y_task.replace({'Shrinkage': 'NonFraud',
                                     'ShrinkageCorrection': 'NonFraud'})
        else:
            y_task = dataset.preprocessed_data[f"y_{split}"]

        if algorithm == 'Autoencoder':
            score_dict = detector.test(data=X_task, device=params.get('device', 'cpu'), return_metrics=False)
            scores = -1 * score_dict['pred']
            evaluation_dict = get_score_dict(scores=scores, y_true=y_task)
        else:
            scores, evaluation_dict = evaluate_detector(detector=detector, X=X_task, y=y_task)
            scores = scores.values

        evaluation_dict = {key + f'_{split}': val for key, val in evaluation_dict.items()}
        out_dict.update(evaluation_dict)
        out_scores.update({f'scores_{split}': scores, f'y_{split}': y_task})

    out_df = pd.DataFrame()
    out_df = out_df.append(out_dict, ignore_index=True)
    out_df.to_csv(os.path.join('./outputs/', experiment_name + '.csv'), index=False)
    print(out_df)
    if output_scores:
        score_df = pd.concat([pd.Series(ndarr, name=name) for name, ndarr in out_scores.items()], axis=1)
        score_df.to_csv(os.path.join('./outputs/', experiment_name + '_scores.csv'), index=False)

    if save_path:
        if algorithm in ['Autoencoder', 'NALU', 'VAE']:
            detector.save(save_path=save_path)
        elif algorithm in ['OneClassSVM']:
            import joblib
            joblib.dump(detector, f'{save_path}.pkl')


if __name__ == '__main__':
    """
    Argparser needs to accept all possible param_search arguments, but only passes given args to params.
    """
    str_args = ('env', 'experiment_name', 'algorithm', 'kernel', 'save_path', 'numeric', 'categorical')
    float_args = ('tol', 'nu', 'max_samples', 'max_features', 'coef0', 'gamma', 'alpha', 'learning_rate')
    int_args = ('n_estimators', 'n_jobs', 'random_state', 'degree', 'cpus',
                'n_neighbors', 'leaf_size',
                'n_layers', 'n_bottleneck', 'epochs', 'batch_size', 'verbose', 'seed', 'cache_size', 'max_iter')
    bool_args = ('bootstrap', 'novelty', 'shuffle', 'output_scores', 'shrinking', 'k', 'numeric_nan_bucket', 'batch_norm')
    parser = ArgumentParser()
    for arg in str_args:
        parser.add_argument(f'--{arg}')
    for arg in int_args:
        parser.add_argument(f'--{arg}', type=int)
    for arg in float_args:
        parser.add_argument(f'--{arg}', type=float)
    for arg in bool_args:
        parser.add_argument(f'--{arg}', action='store_true')
    args_dict = vars(parser.parse_args())

    env = args_dict.pop('env')
    algorithm = args_dict.pop('algorithm')
    experiment_name = args_dict.pop('experiment_name')
    output_scores = args_dict.pop('output_scores')
    numeric = args_dict.pop('numeric')
    categorical = args_dict.pop('categorical')
    numeric_nan_bucket = args_dict.pop('numeric_nan_bucket')
    seed = args_dict.pop('seed')
    np.random.seed(seed)

    erpClassParams = {'train_path': './data/erp_mas/normal/mas_scm_data_fraud3.csv',  # train
                      'eval_path': None,  # eval
                      'test_path': './data/erp_mas/fraud/mas_scm_data_fraud3.csv',  # test
                      'info_path': './data/erp_fraud/column_information.csv',  # column names and dtypes
                      'split_id': None,  # split index
                      }

    params = {key: val for key, val in args_dict.items() if val}  # remove entries with None values
    detect_anomalies(algorithm=algorithm,
                     **erpClassParams,
                     experiment_name=experiment_name,
                     categorical=categorical,
                     numeric=numeric,
                     numeric_nan_bucket=numeric_nan_bucket,
                     params=params,
                     output_scores=output_scores,
                     seed=seed)
