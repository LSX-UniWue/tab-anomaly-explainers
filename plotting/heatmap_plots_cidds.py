
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform

from cidds_xai import xai_to_categorical, get_expl_scores


if __name__ == '__main__':

    approach = 'AE'  # 'AE', 'IF', 'OCSVM'
    save = False

    folder = Path(f'../outputs/explanation/cidds/reproduced/{approach}/')
    for expl_file in folder.glob('*.csv'):
        expl_file = expl_file.stem

        expl = pd.read_csv(folder / f'{expl_file}.csv', header=0, index_col=0)
        if 'expected_value' in expl.columns:
            expl = expl.drop('expected_value', axis=1)

        expl = xai_to_categorical(expl)

        # sanity check
        gold_expl = pd.read_csv(f'../data/cidds/data_raw/anomalies_rand_expl.csv', header=0, index_col=0, encoding='UTF8')
        gold_expl = gold_expl.drop(['attackType', 'label'], axis=1)
        gold_expl = (gold_expl == 'X')
        roc_mean, roc_std = get_expl_scores(explanation=expl, gold_standard=gold_expl, score_type='auc_roc', to_categorical=False)
        cos_mean, cos_std = get_expl_scores(explanation=expl, gold_standard=gold_expl, score_type='cosine_sim', to_categorical=False)
        print(f'ROC AUC: {roc_mean:.3f} +- {roc_std:.3f}')
        print(f'Cosine similarity: {cos_mean:.3f} +- {cos_std:.3f}')

        if roc_mean < 0.5:
            expl = -1 * expl
            roc_mean, roc_std = get_expl_scores(explanation=expl, gold_standard=gold_expl, score_type='auc_roc', to_categorical=False)
            cos_mean, cos_std = get_expl_scores(explanation=expl, gold_standard=gold_expl, score_type='cosine_sim', to_categorical=False)
            print(f'Corrected ROC AUC: {roc_mean:.3f} +- {roc_std:.3f}')
            print(f'Corrected Cosine similarity: {cos_mean:.3f} +- {cos_std:.3f}')


        # sort each attack type by Traffic Protocol + Src IP + Daytime
        indices = [5700910, 6174922, 2155293, 3586426, 3775963, 4104668, 4145062, 5670824, 5966303, 6132132, 2127024,
                   2153066, 3712176, 3789134, 3846555, 745441, 4194894, 4199142, 7893408, 8038356, 7605747, 2616744,
                   3882655, 3924447, 73952, 38098, 6907016, 6041159, 7618576, 7627610, 3923921, 2297603, 3948191,
                   3950385, 7858359, 4524899, 2697931, 4652148, 4737019, 1005858, 2174675, 5603982, 5613953, 7592413,
                   6254157, 7869903, 2550322, 2554517, 2557608, 5483620, 5483682, 5483980, 7584860, 2174317, 7592334,
                   6247567, 6249076, 2175096, 7651715, 2532800, 6240495, 5633456, 7676959, 7679429, 7680271, 7680533,
                   5633445, 5764890, 5781353, 5786928, 5787895, 5801585, 5803587, 5829309, 5865576, 6057089, 6235093,
                   6238585, 7678861, 7679853]
        expl.index = gold_expl.index
        # gold_data = pd.read_csv(f'../data/cidds/data_raw/anomalies_rand_sorted.csv', header=0, index_col=0, encoding='UTF8')
        # gold_data = gold_data.loc[indices]
        data_filtered = expl.apply(lambda row: row > 0.25 * row.max(), axis=1).loc[indices]
        # data_filtered = gold_expl.loc[indices]
        dists = pdist(data_filtered, metric='hamming')
        heatmap = squareform(dists)

        # plot
        font = {'family': 'arial',
                'size': 28}
        matplotlib.rc('font', **font)
        matplotlib.rc('text', usetex=True)
        fig, ax = plt.subplots()
        ax = sns.heatmap(heatmap)

        # axis labels
        ticks = [0, 20, 40, 60, 80]
        label_pos_x = [0.5, 0.5, 0.5, 0.5, 0]
        label_pos_y = [-0.5, -0.5, -0.5, -0.5, 0]

        plt.xticks(ticks=ticks, rotation=0)
        ax.set_xticklabels(['dos', 'port', 'ping', 'brute', ''])
        for i, label in enumerate(ax.xaxis.get_majorticklabels()):
            dx = label_pos_x[i]
            offset = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)

        plt.yticks(ticks=ticks, rotation=0)
        ax.set_yticklabels(['dos', 'port', 'ping', 'brute', ''])
        for i, label in enumerate(ax.yaxis.get_majorticklabels()):
            dy = label_pos_y[i]
            offset = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
            label.set_transform(label.get_transform() + offset)

        plt.tight_layout()
        if save:
            # plt.savefig(f'ground_truth_cidds.png', dpi=400)
            plt.savefig(f'../outputs/explanation/cidds/reproduced/{approach}/{expl_file}.png', dpi=400)
        plt.show()
