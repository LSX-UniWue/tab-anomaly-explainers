
import os
import pickle
import sys
import functools
from pathlib import Path
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import wandb
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics.pairwise import cosine_similarity

from xai.util import xai_to_categorical, tabular_reference_points
from outputs.models.erp_fraud.util import load_best_detector
from data.erp_fraud.erpDataset import ERPDataset
from anomaly_detection.autoencoder_torch import Autoencoder


def get_expl_scores(explanation, gold_standard, dataset, score_type='auc_pr'):
    """Calculate AUC-ROC score for each sample individually, report mean and std"""
    scores = []
    for i, row in explanation.iterrows():
        # Explanation values for each feature treated as likelihood of anomalous feature
        #  -aggregated to feature-scores over all feature assignments
        #  -flattened to match shape of y_true
        #  -inverted, so higher score means more anomalous
        y_score = xai_to_categorical(expl_df=pd.DataFrame(explanation.loc[i]).T,
                                     dataset=dataset).values.flatten() * -1
        # Calculate score
        if score_type == 'auc_roc':
            scores.append(roc_auc_score(y_true=gold_standard.loc[i], y_score=y_score))
        elif score_type == 'auc_pr':
            scores.append(average_precision_score(y_true=gold_standard.loc[i], y_score=y_score))
        elif score_type == 'pearson_corr':
            scores.append(pearsonr(x=gold_standard.loc[i], y=y_score))
        elif score_type == 'cosine_sim':
            scores.append(cosine_similarity(gold_standard.loc[i].values.reshape(1, -1), y_score.reshape(1, -1))[0, 0])
        else:
            raise ValueError(f"Unknown score_type '{score_type}'")

    return np.mean(scores), np.std(scores)


def evaluate_expls(background,
                   train_path,
                   test_path,
                   gold_standard_path,
                   expl_folder,
                   xai_type,
                   job_name,
                   out_path,
                   data):
    """Calculate AUC-ROC score of highlighted important features"""
    expl = pd.read_csv(Path(expl_folder) / f'{xai_type}_{background}_{Path(test_path).stem}_{job_name}.csv',
                       header=0, index_col=0)
    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    # Load gold standard explanations and convert to pd.Series containing
    # expl = expl * -1
    # anomaly index & list of suspicious col names as values
    gold_expl = pd.read_csv(gold_standard_path, header=0, index_col=0, encoding='UTF8')
    gold_expl = (gold_expl == 'X').iloc[:, :-5]  # .apply(lambda x: list(x[x.values].index.values), axis=1)
    to_check = data.get_frauds().index.tolist()

    assert len(to_check) == gold_expl.shape[0], \
        f"Not all anomalies found in explanation: Expected {gold_expl.shape[0]} but got {len(to_check)}"

    # # what if expl is random uniform noise?
    # noise = pd.DataFrame(np.random.uniform(low=0, high=1, size=expl.loc[to_check].shape),
    #                      columns=expl.columns,
    #                      index=expl.loc[to_check].index)
    # # what if expl is random uniform noise, multiplied by input?
    # noise = noise * data.preprocessed_data['X_test'].iloc[to_check]

    # watch out for naming inconsistency! The dataset=data that get_expl_scores gets is an ERPDataset instance!
    roc_mean, roc_std = get_expl_scores(explanation=expl.loc[to_check],
                                        gold_standard=gold_expl.loc[to_check],
                                        score_type='auc_roc',
                                        dataset=data)
    cos_mean, cos_std = get_expl_scores(explanation=expl.loc[to_check],
                                        gold_standard=gold_expl.loc[to_check],
                                        score_type='cosine_sim',
                                        dataset=data)
    pearson_mean, pearson_std = get_expl_scores(explanation=expl.loc[to_check],
                                                gold_standard=gold_expl.loc[to_check],
                                                score_type='pearson_corr',
                                                dataset=data)

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


def explain_anomalies(train_path,
                      test_path,
                      compare_with_gold_standard,
                      expl_folder,
                      job_name,
                      xai_type='shap',
                      model='AE',
                      numeric_preprocessing='bucket',
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
    :param reference:       Option for chosen reference value, used by Integrated Gradients:
                            'nearest-neighbor':         Use nearest neighbor in (normal) training data as reference
                            'optimized':                Use Takeishi optimization to generate normal datapoint in proximity
                            'optimized_adversarial':    Same as optimized, but additionally checks to avoid adversarial samples
    :param compare_with_gold_standard:  Whether or not to evaluate the explanations vs the gold standard
    :param expl_folder:     Str path to folder to write/read explanations to/from
    :param model:           Str type of model to load, one of ['AE', 'OCSVM']
    :param numeric_preprocessing:   Str type of numeric preprocessing, one of ['buckets', 'log10', 'zscore', 'None']
    :param background:      Option for background generation: May be one of:
                            'zeros':                Zero vector as background
                            'mean':                 Takes mean of X_train data through k-means (analog to SHAP)
                            'booking_key_mean':     Takes all X_train data with booking key fitting to samples
                            'booking_key_medoid':   Runs kmedoid with k=1 on all X_train data with fitting booking key
                            'booking_key_NN':       Finds nearest neighbor in X_train data with fitting booking key
                            'NN':                   Finds nearest neighbor in X_train
                            'optimized':            Optimizes samples while keeping one input fixed
                            'full':                 Optimizes every individual sample
    :param kwargs:          Additional keyword args directly for numeric preprocessors during data loading
    """

    data = ERPDataset(train_path=train_path,
                      test_path=test_path,
                      numeric_preprocessing=numeric_preprocessing,
                      categorical_preprocessing='ordinal' if 'ordinal' in xai_type else 'onehot',
                      keep_index=True,
                      **kwargs)

    X_train, _, _, _, X_test, y_test, _, _ = data.preprocessed_data.values()

    # find gold standard explanations for anomalous cases
    ds_file = Path(test_path).stem + "_expls.csv"
    gold_expl_path = f'data/erp_mas/fraud/{ds_file}' if 'mas_scm' in ds_file else f'data/erp_fraud/{ds_file}'
    X_expl = data.get_frauds().sort_index()

    # shard data for running multiple scripts in parallel
    if shard_data is not None:
        i_start = points_per_shard * shard_data
        i_end = min(i_start + points_per_shard, X_expl.shape[0])
        X_expl = X_expl[i_start:i_end]

    print('Loading detector...')
    if rashomon_id is None:
        detector = load_best_detector(model=model, train_path=train_path, test_path=test_path)
    else:
        detector = Autoencoder(**pickle.load(open(Path('./outputs/models/erp_fraud/best_params/AE_session2.p'), 'rb')))
        detector = detector.load(Path('./outputs/models/erp_fraud/AE_rashomon') / f'AE_seed_{rashomon_id}', only_model=False)

    if model in ['IF', 'PCA']:
        detector.fit(X_train)
    if model == 'AE_batch_norm':
        detector.eval()

    # Generating explanations
    if not os.path.exists(os.path.join(expl_folder, f'{xai_type}_{background}_{Path(test_path).stem}_{job_name}.csv')):
        print("Generating explanations...")
        if shard_data is not None:
            out_template = os.path.join(expl_folder, f'{model}_{{}}_{background}_{str(shard_data)}_{Path(test_path).stem}_{job_name}.csv')
        else:
            out_template = os.path.join(expl_folder, f'{{}}_{background}_{Path(test_path).stem}_{job_name}.csv')

        if xai_type in ['shap']:
            import xai.xai_shap

            # set the prediction function
            predict_fn = detector.score_samples

            if background in ['zeros', 'mean', 'NN', 'single_optimum', 'kmeans']:
                # detector.to('cpu')
                reference_points = tabular_reference_points(background=background,
                                                            X_expl=X_expl.values,
                                                            X_train=X_train.values,
                                                            columns=X_expl.columns,
                                                            predict_fn=functools.partial(detector.score_samples,
                                                                                         output_to_numpy=False,
                                                                                         invert_score=False))
            else:
                reference_points = X_train

            xai.xai_shap.explain_anomalies(X_anomalous=X_expl,
                                           predict_fn=predict_fn,
                                           X_benign=reference_points,
                                           background=background,
                                           model_to_optimize=detector,
                                           out_file_path=out_template.format(xai_type))

        elif xai_type in ['lime']:
            import xai.xai_lime
            xai.xai_lime.explain_anomalies(X_anomalous=X_expl,
                                           X_benign=X_train,
                                           xai_type=xai_type,
                                           detector=detector,
                                           out_path=out_template.format(xai_type),
                                           dataset=data,
                                           explainer_seed=explainer_seed,
                                           **kwargs)

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
                                             out_path=out_template.format(xai_type),
                                             target=None,
                                             device='cpu')
        elif xai_type in ['reconstruction']:
            recon = detector.reconstruct(x=X_expl)
            error = -1 * (recon - X_expl)**2
            expl = pd.DataFrame(error, columns=X_expl.columns, index=X_expl.index)
            expl.to_csv(out_template.format(xai_type))

        elif xai_type == 'uniform_noise':
            expl = pd.DataFrame(np.random.rand(*X_expl.shape) * 2 - 1, columns=X_expl.columns, index=X_expl.index)
            expl.to_csv(out_template.format(xai_type))

        elif xai_type == 'uniform_noise_times_input':
            expl = pd.DataFrame(np.random.rand(*X_expl.shape) * 2 - 1, columns=X_expl.columns, index=X_expl.index)
            expl = expl * X_expl
            expl.to_csv(out_template.format(xai_type))

        else:
            raise ValueError(f'Unknown xai_type: {xai_type}')

    if compare_with_gold_standard and shard_data is None:
        print('Evaluating explanations...')
        out_dict = evaluate_expls(background=background,
                                  train_path=train_path,
                                  test_path=test_path,
                                  gold_standard_path=gold_expl_path,
                                  expl_folder=expl_folder,
                                  xai_type=xai_type,
                                  job_name=job_name,
                                  out_path=out_path,
                                  data=data)  # ERPDataset class instance
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

    # ['captum_intgrad', 'captum_lrp', 'captum_gradient', 'captum_grad_input',
    # 'shap', 'lime', 'reconstruction', 'uniform_noise', 'uniform_noise_times_input']
    xai_type = 'shap'

    # ['zeros', 'mean', 'NN', 'kmeans', 'single_optimum', 'optimized']
    background = 'zeros'

    model = 'AE'  # ['AE', 'IF', 'PCA', 'OCSVM']
    # exp setting
    train_path = './data/erp_fraud/normal_2.csv'
    test_path = './data/erp_fraud/fraud_3.csv'
    # mas setting mas data
    # train_path = './data/erp_mas/normal/mas_scm_data_fraud3.csv'
    # test_path = './data/erp_mas/fraud/mas_scm_data_fraud3.csv'
    # mas setting erpsim data
    # train_path = './data/erp_fraud/fraud_3.csv'
    # test_path = './data/erp_fraud/fraud_3.csv'

    compare_with_gold_standard = True
    add_to_summary = False
    out_path = './outputs/explanation/summary.csv' if add_to_summary else None

    # base experiments
    expl_folder = f'./outputs/explanation/erp_fraud/{xai_type}_{background}'
    os.makedirs(expl_folder, exist_ok=True)
    explain_anomalies(train_path=train_path,
                      test_path=test_path,
                      compare_with_gold_standard=compare_with_gold_standard,
                      expl_folder=expl_folder,
                      xai_type=xai_type,
                      model=model,
                      numeric_preprocessing='buckets',
                      background=background,
                      job_name=job_name,
                      shard_data=shard_data,
                      points_per_shard=points_per_shard,
                      out_path=out_path)

    # Rashomon implementation invariance experiments
    # expl_folder = f'./outputs/explanation/erp_fraud/rashomon/{xai_type}_{background}'
    # os.makedirs(expl_folder, exist_ok=True)
    # seeds = [0.0, 11.0, 13.0, 15.0, 16.0, 19.0, 20.0, 22.0, 25.0, 27.0, 28.0, 31.0, 32.0, 37.0, 38.0, 40.0,
    #          41.0, 42.0, 44.0, 4.0, 50.0, 51.0, 54.0, 56.0, 58.0, 5.0, 63.0, 68.0, 69.0, 6.0, 70.0, 71.0, 73.0,
    #          74.0, 76.0, 77.0, 78.0, 79.0, 80.0, 81.0, 82.0, 83.0, 85.0, 87.0, 8.0, 91.0, 92.0, 94.0, 96.0, 97.0]
    # for seed in seeds:
    #     seed = int(seed)
    #     job_name = f'rashomon_{seed}'
    #     explain_anomalies(train_path=train_path,
    #                       test_path=test_path,
    #                       compare_with_gold_standard=False,
    #                       expl_folder=expl_folder,
    #                       xai_type=xai_type,
    #                       model=model,
    #                       numeric_preprocessing='buckets',
    #                       background=background,
    #                       job_name=job_name,
    #                       shard_data=shard_data,
    #                       points_per_shard=points_per_shard,
    #                       rashomon_id=seed,
    #                       out_path=out_path)
