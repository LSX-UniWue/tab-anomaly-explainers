
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from erp_xai import get_expl_scores
from data.erp_fraud.erpDataset import ERPDataset
from xai.util import xai_to_categorical, gini_index, distraction_weight

"""
Evaluate compactness of explanations through Gini Index and weight that does not match ground truth
"""


def run_dist_eval(expl_path, erp_dataset):
    gold_expl = pd.read_csv(f'data/erp_fraud/fraud_3_expls.csv', header=0, index_col=0, encoding='UTF8')
    gold_expl = (gold_expl == 'X').iloc[:, :-5]  # .apply(lambda x: list(x[x.values].index.values), axis=1)
    to_check = erp_dataset.get_frauds().index.tolist()
    assert len(to_check) == gold_expl.shape[0], \
        f"Not all anomalies found in explanation: Expected {gold_expl.shape[0]} but got {len(to_check)}"

    # load all explanations
    expl = pd.read_csv(expl_path, header=0, index_col=0)
    if 'expected_value' in expl.columns:
        expl = expl.drop('expected_value', axis=1)
    assert expl.loc[to_check].shape[0] == gold_expl.loc[to_check].shape[0] == gold_expl.shape[0]
    roc_mean, roc_std = get_expl_scores(explanation=expl, gold_standard=gold_expl, dataset=erp_dataset, score_type='auc_roc')
    print(f"ROC AUC: {roc_mean:.3f} ± {roc_std:.3f}")
    if roc_mean >= 0.5:  # get_expl_scores flips already
        expl = -1 * expl
    expl = xai_to_categorical(expl_df=expl, dataset=erp_dataset)

    rocs = []
    coss = []
    ginis = []
    ginis_abs = []
    distractions = []
    distractions_abs = []
    for i, row in expl.iterrows():
        # Calculate score
        rocs.append(roc_auc_score(y_true=gold_expl.loc[i], y_score=row))
        coss.append(cosine_similarity(gold_expl.loc[i].values.reshape(1, -1), row.values.reshape(1, -1))[0, 0])
        ginis.append(gini_index(row=row))
        ginis_abs.append(gini_index(row=row, absolute=True))
        distractions.append(distraction_weight(expl=row, gold=gold_expl.loc[i]))
        distractions_abs.append(distraction_weight(expl=row, gold=gold_expl.loc[i], absolute=True))
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

    # Save results
    out_path = expl_path.parent / 'summary.csv'
    if not out_path.exists():
        pd.DataFrame(columns=res_dict.keys()).to_csv(out_path, index=False)
    pd.read_csv(out_path).append(res_dict, ignore_index=True).to_csv(out_path, index=False)


if __name__ == '__main__':
    in_path = Path('outputs/explanation/erp_fraud/reproduced/OCSVM')
    data = ERPDataset(train_path='./data/erp_fraud/normal_2.csv',
                      test_path='./data/erp_fraud/fraud_3.csv',
                      numeric_preprocessing='buckets',
                      categorical_preprocessing='onehot',
                      keep_index=True)
    # run_dist_eval(in_path / 'shap_mean_normal_2_fraud_3.csv', erp_dataset=data)
    for expl_path in in_path.glob('*.csv'):
        if expl_path.stem == 'summary':
            continue
        run_dist_eval(expl_path, erp_dataset=data)
