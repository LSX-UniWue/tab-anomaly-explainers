
import os
import pickle as pkl
from pathlib import Path
from joblib import parallel_backend, Parallel, delayed

import torch
import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib

from rashomon_util import _explanation_distance, _top_k_eucl, _top_k_sign_agreement, _top_k_feature_agreement
from cidds_xai import get_expl_scores, xai_to_categorical


def get_mean_scores():
    base_folder = Path('outputs/explanation/cidds/rashomon')
    invert_dict_cidds = {'captum_grad_input_zeros': True,
                         'captum_gradient_zeros': False,
                         'captum_intgrad_mean': False,
                         'captum_intgrad_NN': False,
                         'captum_intgrad_single_optimum': False,
                         'captum_intgrad_zeros': False,
                         'captum_lrp_grad': True,
                         'captum_lrp_nograd': True,
                         'lime_kmeans': True,
                         'shap_kmeans': True,
                         'shap_mean': True,
                         'shap_NN': True,
                         'shap_single_optimum': True,
                         'shap_zeros': True}

    # evaluate explanations to get direction of scores
    gold_expl = pd.read_csv('data/cidds/data_raw/anomalies_rand_expl.csv', header=0, index_col=0, encoding='UTF8')
    gold_expl = gold_expl.drop(['attackType', 'label'], axis=1)
    gold_expl = (gold_expl == 'X')

    for in_folder in base_folder.iterdir():
        if not in_folder.is_dir():
            continue
        print(f"Processing {in_folder.name}")
        expl_files = list(in_folder.glob('*.csv'))
        outs = []
        for expl_file in tqdm.tqdm(expl_files):

            expl = pd.read_csv(expl_file, header=0, index_col=0)
            if 'expected_value' in expl.columns:
                expl = expl.drop('expected_value', axis=1)
            assert expl.shape[0] == gold_expl.shape[0]

            roc_mean, roc_std = get_expl_scores(explanation=expl,
                                                gold_standard=gold_expl,
                                                score_type='auc_roc')
            cos_mean, cos_std = get_expl_scores(explanation=expl,
                                                gold_standard=gold_expl,
                                                score_type='cosine_sim')

            res_dict = {
                'seed': int(expl_file.stem.split('_')[-1]),
                'roc_mean': roc_mean,
                'cos_mean': cos_mean,
            }
            outs.append(res_dict)

        out_dict = {'type': 'large',
                    'Model': in_folder.name,
                    'gte 0.5': len([1 for x in outs if x['roc_mean'] >= 0.5]),
                    'lt 0.5': len([1 for x in outs if x['roc_mean'] < 0.5]),
                    'Mean mean ROCs': np.mean([x['roc_mean'] for x in outs]),
                    'Std mean ROCs': np.std([x['roc_mean'] for x in outs]),
                    'Mean mean COSs': np.mean([x['cos_mean'] for x in outs]),
                    'Std mean COSs': np.std([x['cos_mean'] for x in outs])}

        out_file = base_folder / 'summary.csv'
        if out_file.exists():
            summary = pd.read_csv(out_file, header=0)
            summary = summary.append(out_dict, ignore_index=True)
        else:
            summary = pd.DataFrame(out_dict, index=[0])
        summary.to_csv(out_file, index=False)


def hist_plots(in_folder, _k=3, type='large', just_legend=False):
    if just_legend:
        plt.figure(figsize=(4, 1))
        matplotlib.rcParams['text.usetex'] = True
        matplotlib.rcParams.update({'font.size': 12})
        colors = ['red', 'orange', 'green', 'blue']
        metric_names = ['euclid abs', 'euclid', 'sign disagreement', 'feature disagreement']
        for (metric_name, color) in zip(metric_names, colors):
            plt.hist([], label=metric_name, color=color, lw=2, alpha=.15)
        # legend in two columns
        plt.legend(loc='center', ncol=2)
        plt.axis('off')
        plt.savefig(in_folder.parent / 'histograms' / 'legend.pdf', bbox_inches='tight')
        plt.show()
        return

    # load expl roc scores
    roc_scores = pd.read_csv(in_folder.parent / 'summary.csv')
    roc_scores = roc_scores[roc_scores['type'] == type]
    mean_roc = roc_scores[roc_scores['Model'] == in_folder.name]['Mean mean ROCs'].values[0]
    std_roc = roc_scores[roc_scores['Model'] == in_folder.name]['Std mean ROCs'].values[0]
    flip_scores = mean_roc < 0.5
    if flip_scores:
        mean_roc = 1. - mean_roc

    # load all explanations
    model_dfs = []
    for expl_file in in_folder.glob('*.csv'):
        seed = int(expl_file.stem.split('_')[-1])
        if type == 'small' and seed not in [19, 25, 27, 32, 38, 51, 5, 71, 74, 80, 83]:
            continue
        expl = pd.read_csv(expl_file, header=0, index_col=0)
        if 'expected_value' in expl.columns:
            expl = expl.drop('expected_value', axis=1)
        if flip_scores:
            expl = -1 * expl
        expl = xai_to_categorical(expl)
        # normalize all datapoints with l2 norm except zero vector
        l2_norm = np.linalg.norm(expl, axis=1, ord=2)[:, None]
        l2_norm[l2_norm == 0] = 1
        expl = expl / l2_norm
        model_dfs.append(expl)

    # reform to shape=(n_models, n_samples, dim_data)
    _expls_stacked = np.stack([df.values for df in model_dfs], axis=0)

    # compute dissimilarity scores for all pairs of explanations
    metrics = [('euclid abs', lambda e1, e2: _explanation_distance(np.abs(e1), np.abs(e2), metric=_top_k_eucl, k=1.)),
               ('euclid', lambda e1, e2: _explanation_distance(e1, e2, metric=_top_k_eucl, k=1.)),
               ('sign disagreement', lambda e1, e2: 1. - _explanation_distance(e1, e2, metric=_top_k_sign_agreement, k=_k)),
               ('feature disagreement', lambda e1, e2: 1. - _explanation_distance(e1, e2, metric=_top_k_feature_agreement, k=_k))]

    dists = []
    with parallel_backend(backend='loky', n_jobs=8):
        # pre_compute masks because they will not differ for explanations and would be computed multiple
        # times in Parallel() clause below

        for (_, metric) in tqdm.tqdm(metrics, position=1, desc="distances"):
            # compute upper triangle of full distance matrix
            _dists = Parallel(verbose=1)(delayed(metric)
                                         (_expls_stacked[i, :],
                                          _expls_stacked[j, :])
                                         for i in range(_expls_stacked.shape[0])
                                         for j in range(i + 1, _expls_stacked.shape[0]))
            dists.append(_dists)

    # plot histograms
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.preamble'] = [r'\usepackage{amsmath}']
    matplotlib.rcParams.update({'font.size': 12})
    colors = ['red', 'orange', 'green', 'blue']
    kwargs_outline = dict(histtype='step', alpha=1., density=True, bins=75)
    kwargs_filled = dict(histtype='stepfilled', alpha=0.15, density=True, bins=75)
    plt.xlim(0, 1)
    # remove axis labels and ticks
    plt.gca().axes.get_xaxis().set_ticks([])
    plt.gca().axes.get_yaxis().set_ticks([])
    plt.gca().axes.get_xaxis().set_ticklabels([])
    plt.gca().axes.get_yaxis().set_ticklabels([])
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    # remove box around plot, only take x axis
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

    for (metric_name, _dists, color) in zip([m[0] for m in metrics], dists, colors):
        if 'eucl' in metric_name:
            if np.max(_dists) > 1.:
                _dists = _dists / np.max(_dists)
        plt.hist(_dists, **kwargs_filled, color=color, label=metric_name)  # plot transparent filling
        plt.hist(_dists, **kwargs_outline, color=color, lw=0.5)  # plot full colored outline

    # add in_folder name as title and mean roc scores as subtitle
    names = {'captum_grad_input_zeros': 'Gradient×Input',
             'captum_gradient_zeros': 'Saliency',
             'captum_intgrad_mean': 'Integrated Gradients (mean)',
             'captum_intgrad_NN': 'Integrated Gradients (NN)',
             'captum_intgrad_single_optimum': 'Integrated Gradients (lopt)',
             'captum_intgrad_zeros': 'Integrated Gradients (zeros)',
             'captum_lrp_grad': 'LRP (with grad)',
             'captum_lrp_nograd': 'LRP (no grad)',
             'lime_kmeans': 'LIME (kmeans)',
             'shap_kmeans': 'SHAP (kmeans)',
             'shap_mean': 'SHAP (mean)',
             'shap_NN': 'SHAP (NN)',
             'shap_single_optimum': 'SHAP (lopt)',
             'shap_zeros': 'SHAP (zeros)'}
    plt.title("ROC$_{XAI}$ scores across models: " + f"${mean_roc:.2f} \pm {std_roc:.2f}$")
    plt.suptitle(f"{names[in_folder.name]}")
    # save to pdf
    plt.savefig(in_folder.parent / 'histograms' / f'{in_folder.name}_{type}.pdf', bbox_inches='tight')
    plt.show()


if __name__ == '__main__':
    # important notes
    #  - k: min is 3 in ground truth, mean is 3.7125 -> 3 it is
    #  - we run an l2 norm since the euclidean distance is sensitive to value range (all other metrics arent)
    #  - we aggregate scores first
    score_path = Path('outputs/explanation/cidds/rashomon')
    for in_folder in score_path.iterdir():
        if in_folder.is_dir() and not in_folder.name == 'histograms':
            hist_plots(in_folder=in_folder)
    hist_plots(in_folder=score_path / 'histograms', just_legend=True)
