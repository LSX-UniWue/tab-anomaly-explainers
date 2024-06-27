
import os
import sys

import functools
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity
import dask.array as da
from dask_ml.wrappers import ParallelPostFit

from xai.util import tabular_reference_points
from data.cidds.util import get_cols_and_dtypes, get_column_mapping, get_summed_columns


class DaskOCSVM:
    """Small wrapper to trick dask_ml into parallelizing anomaly detection methods"""

    def __init__(self, model):
        self.model = model

    def predict(self, X):
        return self.model.score_samples(X)


def xai_to_categorical(df, cat_encoding='onehot', num_encoding='quantized'):
    # sum all encoded scores to single categorical values for each column
    categories = get_column_mapping(cat_encoding=cat_encoding, num_encoding=num_encoding, as_int=True)
    category_names = get_summed_columns()

    data = df.values
    data_cat = np.zeros((data.shape[0], len(categories)))
    for i, cat in enumerate(categories):
        data_cat[:, i] = np.sum(data[:, cat], axis=1)
    data_cat = pd.DataFrame(data_cat, columns=category_names, index=df.index)
    return data_cat


def get_expl_scores(explanation, gold_standard, score_type='auc_roc', to_categorical=True):
    """Calculate AUC-ROC score for each sample individually, report mean and std"""
    # Explanation values for each feature treated as likelihood of anomalous feature
    #  -aggregated to feature-scores over all feature assignments
    #  -flattened to match shape of y_true
    #  -inverted, so higher score means more anomalous
    if to_categorical:
        explanation = xai_to_categorical(explanation)
    scores = []
    for i, row in explanation.iterrows():
        # Calculate score
        if score_type == 'auc_roc':
            scores.append(roc_auc_score(y_true=gold_standard.iloc[i], y_score=row))
        elif score_type == 'auc_pr':
            scores.append(average_precision_score(y_true=gold_standard.iloc[i], y_score=row))
        elif score_type == 'pearson_corr':
            scores.append(pearsonr(x=gold_standard.iloc[i], y=row))
        elif score_type == 'cosine_sim':
            scores.append(cosine_similarity(gold_standard.iloc[i].values.reshape(1, -1), row.values.reshape(1, -1))[0, 0])
        else:
            raise ValueError(f"Unknown score_type '{score_type}'")

    return np.mean(scores), np.std(scores)


def evaluate_expls(background,
                   model,
                   gold_standard_path,
                   expl_path,
                   xai_type,
                   out_path):
    """Calculate AUC-ROC score of highlighted important features"""
    expl = pd.read_csv(expl_path, header=0, index_col=0)
    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    if model in ['IF', 'OCSVM', 'AE']:  # switch expl sign here
        expl = -1 * expl
    # Load gold standard explanations and convert to pd.Series containing
    # anomaly index & list of suspicious col names as values
    gold_expl = pd.read_csv(gold_standard_path, header=0, index_col=0, encoding='UTF8')
    gold_expl = gold_expl.drop(['attackType', 'label'], axis=1)
    gold_expl = (gold_expl == 'X')

    assert expl.shape[0] == gold_expl.shape[0], \
        f"Not all anomalies found in explanation: Expected {gold_expl.shape[0]} but got {expl.shape[0]}"

    roc_mean, roc_std = get_expl_scores(explanation=expl,
                                        gold_standard=gold_expl,
                                        score_type='auc_roc')
    cos_mean, cos_std = get_expl_scores(explanation=expl,
                                        gold_standard=gold_expl,
                                        score_type='cosine_sim')
    pearson_mean, pearson_std = get_expl_scores(explanation=expl,
                                                gold_standard=gold_expl,
                                                score_type='pearson_corr')

    out_dict = {'xai': xai_type,
                'variant': background,
                f'ROC': roc_mean,
                f'ROC-std': roc_std,
                f'Cos': cos_mean,
                f'Cos-std': cos_std,
                f'Pearson': pearson_mean,
                f'Pearson-std': pearson_std}
    [print(key + ':', val) for key, val in out_dict.items()]

    # save outputs to combined result csv file
    if out_path:
        if os.path.exists(out_path):
            out_df = pd.read_csv(out_path, header=0)
        else:
            out_df = pd.DataFrame()
        out_df = out_df.append(out_dict, ignore_index=True)
        out_df.to_csv(out_path, index=False)
    return out_dict


def explain_anomalies(compare_with_gold_standard,
                      expl_folder,
                      job_name,
                      xai_type='shap',
                      model='AE',
                      background='zeros',
                      shard_data=None,
                      points_per_shard=5,
                      out_path=None,
                      rashomon_id=None,
                      explainer_seed=None,
                      **kwargs):
    """
    :param train_path:      Str path to train dataset
    :param test_path:       Str path to test dataset
    :param expl_folder:     Str path to folder to write/read explanations to/from
    :param model:           Str type of model to load, one of ['AE', 'OCSVM', 'IF']
    :param background:      Option for background generation: May be one of:
                            'zeros':                Zero vector as background
                            'mean':                 Takes mean of X_train data through k-means (analog to SHAP)
                            'NN':                   Finds nearest neighbor in X_train
                            'optimized':            Optimizes samples while keeping one input fixed
                            'full':                 Optimizes every individual sample
    :param kwargs:          Additional keyword args directly for numeric preprocessors during data loading
    """

    print('Loading data...')
    cols, dtypes = get_cols_and_dtypes(cat_encoding='onehot', num_encoding='quantized')
    X_expl = pd.read_csv(Path('.') / 'data' / 'cidds' / 'data_prep' / 'onehot_quantized' / f'anomalies_rand.csv',
                         index_col=None, usecols=cols + ['attackType'], header=0, dtype={'attackType': str, **dtypes})
    y_test = X_expl.pop('attackType')

    # shard data for running multiple scripts in parallel
    if shard_data is not None:
        i_start = points_per_shard * shard_data
        i_end = min(i_start + points_per_shard, X_expl.shape[0])
        X_expl = X_expl[i_start:i_end]

    if background in ['mean', 'kmeans', 'NN']:
        X_train = pd.read_csv(Path('.') / 'data' / 'cidds' / 'data_prep' / 'onehot_quantized' / 'train.csv.gz',
                              index_col=None, usecols=cols, header=0, dtype=dtypes, compression='gzip')
        X_train = X_train.sample(frac=0.001, random_state=42)  # sample normal data for kmeans and NN background
    else:
        X_train = pd.DataFrame(np.empty(X_expl.shape), columns=X_expl.columns, index=X_expl.index)

    print('Loading detector...')
    if model == 'AE':
        from anomaly_detection.autoencoder_torch import Autoencoder
        params = {'cpus': 8, 'n_layers': 3, 'n_bottleneck': 32, 'epochs': 10, 'batch_size': 2048, 'verbose': 2,
                  'learning_rate': 0.01, 'n_inputs': 146}  # best params for cidds-ae-16
        # params = {'cpus': 8, 'n_layers': 3, 'n_bottleneck': 32, 'epochs': 10, 'batch_size': 2048, 'verbose': 2,
        # 'learning_rate': 0.001, 'n_inputs': 146, 'batch_norm': True}  # best params for cidds-ae-16_batch_norm
        detector = Autoencoder(**params)
        if rashomon_id is not None:
            detector.load(f'./outputs/models/cidds/Autoencoder/rashomon_{rashomon_id}_state_dict.pt')
        else:
            detector = detector.load('./outputs/models/cidds/cidds-ae-16_best.pt')
        # detector = detector.load('./outputs/models/cidds/cidds-ae-16_batch_norm.pt')
        detector.to('cpu')
        detector.eval()
    elif model == 'IF':
        import joblib
        detector = joblib.load('./outputs/models/cidds/cidds-if-41_best.pkl')
    elif model == 'OCSVM':
        import joblib
        detector = joblib.load('./outputs/models/cidds/cidds-oc-12_best.pkl')
    else:
        raise ValueError(f"Model {model} not supported!")

    # Generating explanations
    if 'rashomon' in job_name:
        out_path = Path(expl_folder) / 'rashomon' / f'{xai_type}_{background}' / f'{job_name}.csv'
    elif 'numerical' in job_name:
        out_path = Path(expl_folder) / 'numerical' / f'{xai_type}_{background}_{job_name}.csv'
    else:
        out_path = Path(expl_folder) / f'{model}_{xai_type}_{background}_{job_name}.csv'
    if not out_path.exists():
        print("Generating explanations...")
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(detector, ParallelPostFit):  # trick for multiprocessing single core algorithms with dask
            def predict_fn(X):
                data = da.from_array(X, chunks=(100, -1))
                return detector.predict(data).compute()
        else:
            predict_fn = detector.score_samples

        if xai_type == 'shap':
            import xai.xai_shap

            if background in ['zeros', 'mean', 'NN', 'single_optimum', 'kmeans', 'kmedoids']:
                if model == 'AE':
                    ref_predict_fn = functools.partial(detector.score_samples, output_to_numpy=False, invert_score=False)
                else:
                    ref_predict_fn = predict_fn

                reference_points = tabular_reference_points(background=background,
                                                            X_expl=X_expl.values,
                                                            X_train=X_train.values,
                                                            columns=X_expl.columns,
                                                            predict_fn=ref_predict_fn)
            else:
                reference_points = X_train

            xai.xai_shap.explain_anomalies(X_anomalous=X_expl,
                                           predict_fn=predict_fn,
                                           X_benign=reference_points,
                                           background=background,
                                           model_to_optimize=detector,
                                           out_file_path=out_path)

        elif xai_type == 'lime':
            import xai.xai_lime
            xai.xai_lime.explain_anomalies(X_anomalous=X_expl,
                                           X_benign=X_train,
                                           xai_type=xai_type,
                                           detector=detector,
                                           out_path=out_path,
                                           predict_fn=predict_fn,
                                           explainer_seed=explainer_seed,
                                           dataset=None)

        elif xai_type in ['captum_intgrad', 'captum_lrp', 'captum_gradient', 'captum_grad_input']:
            import xai.xai_captum

            if xai_type in ['captum_intgrad']:  # approach needs a single background point per sample to be explained
                reference_points = tabular_reference_points(background=background,
                                                            X_expl=X_expl.values,
                                                            X_train=X_train.values,
                                                            columns=X_expl.columns,
                                                            predict_fn=functools.partial(detector.score_samples,
                                                                                         output_to_numpy=False,
                                                                                         invert_score=False))
            else:
                reference_points = None

            def predict_fn(X, detector):
                y = detector.score_samples(X, output_to_numpy=False, invert_score=False)
                return y

            xai.xai_captum.explain_anomalies(X_anomalous=X_expl,
                                             reference_points=reference_points,
                                             xai_type=xai_type,
                                             model=detector,
                                             predict_fn=functools.partial(predict_fn, detector=detector),
                                             out_path=out_path,
                                             target=None,
                                             device='cpu')

        elif xai_type == 'reconstruction':
            recon = detector.reconstruct(x=X_expl)
            error = (recon - X_expl)**2
            expl = pd.DataFrame(error, columns=X_expl.columns, index=X_expl.index)
            expl.to_csv(out_path)

        elif xai_type == 'uniform_noise':
            expl = pd.DataFrame(np.random.rand(*X_expl.shape) * 2 - 1, columns=X_expl.columns, index=X_expl.index)
            expl.to_csv(out_path)

        elif xai_type == 'uniform_noise_times_input':
            expl = pd.DataFrame(np.random.rand(*X_expl.shape) * 2 - 1, columns=X_expl.columns, index=X_expl.index)
            expl = expl * X_expl
            expl.to_csv(out_path)

        else:
            raise ValueError(f'Unknown xai_type: {xai_type}')

    if compare_with_gold_standard and shard_data is None:
        print('Evaluating explanations...')
        out_dict = evaluate_expls(background=background,
                                  expl_path=f'./outputs/explanation/cidds/{model}_{xai_type}_{background}_{job_name}.csv',
                                  gold_standard_path=f'data/cidds/data_raw/anomalies_rand_expl.csv',
                                  xai_type=xai_type,
                                  model=model,
                                  out_path=out_path)
        return out_dict


if __name__ == '__main__':
    """
    Argparser needs to accept all possible param_search arguments, but only passes given args to params.
    """

    parser = ArgumentParser()
    parser.add_argument(f'--shard_data', type=int, default=None)
    parser.add_argument(f'--points_per_shard', type=int, default=None)
    parser.add_argument(f'--job_name', type=str, default='')
    args_dict = vars(parser.parse_args())
    shard_data = args_dict.pop('shard_data') if 'shard_data' in args_dict else None
    points_per_shard = args_dict.pop('points_per_shard') if 'points_per_shard' in args_dict else None
    job_name = args_dict.pop('job_name') if 'job_name' in args_dict else None

    # works: ['zeros', 'mean', 'kmeans', 'NN', 'optimized']
    # IG @ AE: ['single_optimum']
    background = 'NN'
    model = 'AE'  # ['AE', 'IF', 'OCSVM']
    # ['lime', 'shap', 'captum_gradient', 'captum_grad_input', 'captum_intgrad', 'captum_lrp',
    # 'uniform_noise', 'uniform_noise_times_input', 'reconstruction']
    xai_type = 'shap'
    compare_with_gold_standard = False
    add_to_summary = False
    out_path = './outputs/explanation/cidds_summary.csv' if add_to_summary else None

    # base explanations
    # expl_folder = './outputs/explanation/cidds'
    # explain_anomalies(compare_with_gold_standard=compare_with_gold_standard,
    #                   expl_folder=expl_folder,
    #                   xai_type=xai_type,
    #                   model=model,
    #                   background=background,
    #                   job_name=job_name,
    #                   shard_data=shard_data,
    #                   points_per_shard=points_per_shard,
    #                   out_path=out_path)

    # rashomon implementation invariance experiments
    expl_folder = './outputs/explanation/cidds'
    for seed in range(20):
        job_name = f'rashomon_{seed}'
        explain_anomalies(compare_with_gold_standard=compare_with_gold_standard,
                          expl_folder=expl_folder,
                          xai_type=xai_type,
                          model=model,
                          background=background,
                          job_name=job_name,
                          shard_data=shard_data,
                          points_per_shard=points_per_shard,
                          rashomon_id=seed,
                          out_path=out_path)
