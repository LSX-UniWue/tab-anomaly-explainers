
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from cidds_xai import get_expl_scores, xai_to_categorical
from xai.util import gini_index, distraction_weight


def run_dist_eval(expl_path):
    # evaluate explanations to get direction of scores
    gold_expl = pd.read_csv('data/cidds/data_raw/anomalies_rand_expl.csv', header=0, index_col=0, encoding='UTF8')
    gold_expl = gold_expl.drop(['attackType', 'label'], axis=1)
    gold_expl = (gold_expl == 'X')

    expl = pd.read_csv(expl_path, header=0, index_col=0)
    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    assert expl.shape[0] == gold_expl.shape[0]

    roc_mean, roc_std = get_expl_scores(explanation=expl,
                                        gold_standard=gold_expl,
                                        score_type='auc_roc')
    print(f"ROC AUC: {roc_mean:.3f} ± {roc_std:.3f}")
    if roc_mean < 0.5:
        expl = -1 * expl
    expl = xai_to_categorical(df=expl)

    rocs = []
    coss = []
    ginis = []
    ginis_abs = []
    distractions = []
    distractions_abs = []
    for i, row in expl.iterrows():
        # Calculate score
        rocs.append(roc_auc_score(y_true=gold_expl.iloc[i], y_score=row))
        coss.append(cosine_similarity(gold_expl.iloc[i].values.reshape(1, -1), row.values.reshape(1, -1))[0, 0])
        ginis.append(gini_index(row=row))
        ginis_abs.append(gini_index(row=row, absolute=True))
        distractions.append(distraction_weight(expl=row, gold=gold_expl.iloc[i]))
        distractions_abs.append(distraction_weight(expl=row, gold=gold_expl.iloc[i], absolute=True))

    res_dict = {
        'expl': expl_path.stem,
        'model': expl_path.parent.stem,
        'roc_mean': np.mean(rocs),
        'cos_mean': np.mean(coss),
        'gini_mean': np.mean(ginis),
        'gini_std': np.std(ginis),
        'gini_abs_mean': np.mean(ginis_abs),
        'gini_abs_std': np.std(ginis_abs),
        'dist_mean': np.mean(distractions),
        'dist_std': np.std(distractions),
        'dist_abs_mean': np.mean(distractions_abs),
        'dist_abs_std': np.std(distractions_abs),
    }
    print(res_dict)

    out_file = expl_path.parent / 'summary.csv'
    if not out_file.exists():
        pd.DataFrame(columns=res_dict.keys()).to_csv(out_file, index=False)
    pd.read_csv(out_file).append(res_dict, ignore_index=True).to_csv(out_file, index=False)


if __name__ == '__main__':
    in_path = Path('outputs/explanation/cidds/reproduced/AE')
    # run_dist_eval(in_path / 'shap_mean_normal_2_fraud_3.csv', erp_dataset=data)
    for expl_path in in_path.glob('*.csv'):
        if expl_path.stem == 'summary':
            continue
        run_dist_eval(expl_path)
