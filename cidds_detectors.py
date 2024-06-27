import os
import sys
from contextlib import nullcontext

import joblib

from anomaly_detection.util import TorchRandomSeed

from argparse import ArgumentParser
from pathlib import Path
from collections import ChainMap
import datetime

import numpy as np
import pandas as pd
import dask.array as da
from dask_ml.wrappers import ParallelPostFit
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score, roc_auc_score
import torch.utils.data

from data.cidds.util import get_cols_and_dtypes
from anomaly_detection.autoencoder_torch import Autoencoder


def detect_anomalies(algorithm, params, seed, evaluation_save_path, model_load_path=None, job_name=None, device='cpu'):

    # setup args
    source_path = './'
    num_encoding = 'quantized'
    data_folder = 'onehot_quantized'
    if 'device' in params:
        device = params.pop('device')  # 'cuda', 'cpu'
        print(f'device: {device}')
    cols, dtypes = get_cols_and_dtypes(cat_encoding='onehot', num_encoding=num_encoding)

    if job_name is not None and 'rashomon' in job_name:
        np.random.seed(42)  # rashomon: same seed for everything but model initialization
    else:
        np.random.seed(seed)

    # Load model
    if algorithm == 'IsolationForest':
        detector_class = IsolationForest
        params['random_state'] = seed
    elif algorithm == 'OneClassSVM':
        detector_class = OneClassSVM
        params.pop('seed')
    elif algorithm == 'Autoencoder':
        detector_class = Autoencoder
        params['n_inputs'] = len(cols)
        if job_name is not None and 'rashomon' in job_name:
            try:
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
            except NameError or ModuleNotFoundError:
                pass
    else:
        raise ValueError(f"Variable algorithm was: {algorithm}")

    # Training
    with TorchRandomSeed(seed) if algorithm == 'Autoencoder' and job_name is not None and 'rashomon' in job_name else nullcontext:
        detector = detector_class(**params)
    if model_load_path is not None:
        detector.load(model_load_path)
    else:
        # Load data
        print(f'Loading train ...')
        train = pd.read_csv(Path(source_path) / 'data' / 'cidds' / 'data_prep' / data_folder / 'train.csv.gz',
                            index_col=None, usecols=cols, header=0, dtype=dtypes, compression='gzip')[cols]

        print('Training ...')
        if algorithm == 'IsolationForest':
            detector = detector.fit(train)
        elif algorithm == 'OneClassSVM':
            train = train.sample(frac=0.001, random_state=seed)
            detector = detector.fit(train)
        elif algorithm == 'Autoencoder':
            with TorchRandomSeed(42) if job_name is not None and 'rashomon' in job_name else nullcontext:
                detector = detector.fit(train, device=device)
        del train

        if job_name is not None:
            out_path = Path('./outputs/models/cidds/') / algorithm
            if not out_path.exists():
                out_path.mkdir(parents=True)
            if algorithm in ['IsolationForest', 'OneClassSVM']:
                joblib.dump(detector, out_path / f'{job_name}.pkl')
            elif algorithm == 'Autoencoder':
                detector.save(out_path / f'{job_name}_state_dict.pt')

    # Evaluation
    eval_out = []
    for split in ['valid', 'test']:
        print(f'Loading {split} ...')
        eval_data = pd.read_csv(Path(source_path) / 'data' / 'cidds' / 'data_prep' / data_folder / f'{split}.csv.gz',
                                index_col=None, usecols=cols + ['isNormal'], header=0,
                                dtype={'isNormal': np.int8, **dtypes}, compression='gzip')[cols + ['isNormal']]
        y = 1 - eval_data.pop('isNormal')

        print(f'Evaluating {split} ...')
        if algorithm == 'IsolationForest':
            scores = -1 * pd.Series(detector.score_samples(eval_data), index=eval_data.index)
            out_dict = {f'auc_pr_{split}': average_precision_score(y_true=y, y_score=scores),
                        f'auc_roc_{split}': roc_auc_score(y_true=y, y_score=scores)}
        elif algorithm == 'OneClassSVM':
            scores = -1 * pd.Series(detector.predict(da.from_array(eval_data.values, chunks=(100, -1))).compute(),
                                    index=eval_data.index)
            out_dict = {f'auc_pr_{split}': average_precision_score(y_true=y, y_score=scores),
                        f'auc_roc_{split}': roc_auc_score(y_true=y, y_score=scores)}
        elif algorithm == 'Autoencoder':
            y_eval = torch.tensor(y)
            x = torch.Tensor(eval_data[cols].values)
            eval_data = torch.utils.data.TensorDataset(x, y_eval)
            eval_loader = torch.utils.data.DataLoader(dataset=eval_data, batch_size=params['batch_size'],
                                                      num_workers=0, shuffle=False)
            out_dict = detector.test(eval_loader, device=device)

        out_dict = {key + f'_{split}': val for key, val in out_dict.items()}
        print(out_dict)
        eval_out.append(out_dict)
    # Outputs
    if evaluation_save_path is not None:
        out_dict = dict(ChainMap(*eval_out))
        out_df = pd.DataFrame()
        out_df = out_df.append({**params, **out_dict, 'seed': seed, 'job_name': job_name}, ignore_index=True)
        out_name = job_name if job_name is not None else datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        out_path = Path(evaluation_save_path) / algorithm
        if not out_path.exists():
            out_path.mkdir(parents=True)
        out_df.to_csv(out_path / f'{out_name}.csv', index=False)


if __name__ == '__main__':
    """
    Argparser needs to accept all possible param_search arguments, but only passes given args to params.
    """
    str_args = ('algorithm', 'evaluation_save_path', 'job_name', 'model_load_path', 'device')
    float_args = ('learning_rate', 'max_samples', 'max_features', 'kernel', 'gamma', 'nu', 'tol')
    int_args = ('cpus', 'n_layers', 'n_bottleneck', 'epochs', 'batch_size', 'verbose', 'seed', 'n_estimators',
                'random_state', 'bootstrap', 'n_jobs', 'shrinking', 'cache_size', 'max_iter')
    bool_args = ['batch_norm']
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

    evaluation_save_path = args_dict.pop('evaluation_save_path')
    model_load_path = args_dict.pop('model_load_path')
    job_name = args_dict.pop('job_name')
    algorithm = args_dict.pop('algorithm')
    params = {key: val for key, val in args_dict.items() if val}  # remove entries with None values

    detect_anomalies(algorithm=algorithm,
                     params=params,
                     seed=args_dict['seed'],
                     evaluation_save_path=evaluation_save_path,
                     model_load_path=model_load_path,
                     job_name=job_name)
