
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from data.erp_fraud.erpDataset import ERPDataset
from xai.util import xai_to_categorical


if __name__ == '__main__':

    approach = 'AE'  # 'AE', 'OCSVM', 'IF'
    expl_file = 'captum_grad_input_zeros_fraud_3_'
    save = False

    expl = pd.read_csv(f'../outputs/explanation/erp_fraud/reproduced/{approach}/{expl_file}.csv', index_col=0)
    save_path = f'./figures/heatmap_{expl_file}.png' if save else None
    # Fix order:
    # [Larceny II], [Larceny IV], [Larceny V], [Invoice Kickback I], [Invoice Kickback II], [Corporate Injury],
    # then after occurrence
    fraud_order = [13759, 13760, 13765, 13766, 22205, 22206, 22211, 22212, 2494, 2495, 2496, 2497, 2498,
                   2499, 2500, 2501, 2502, 2503, 8755, 8756, 8757, 8758, 8759, 8760, 8761, 8762, 8763, 8764,
                   8765, 8766, 8767, 8768, 8769, 8770, 8771, 8772, 8773, 8774, 8775, 8776, 35046, 35047,
                   35048, 35049, 684, 685, 5064, 5065, 15745, 15746, 15747, 15748, 15749, 15750, 15751,
                   15752, 15753, 15754, 15755, 15756, 22424, 22425, 22426, 22427, 22428, 22429, 22430,
                   22431, 22432, 22433, 22434, 22435, 361, 362, 1035, 1036, 1037, 1038, 36376, 36377,
                   36378, 36379, 36802, 36803, 36804, 36805]
    data = expl.loc[fraud_order]
    if 'expected_value' in data.columns:
        data = data.drop('expected_value', axis=1)

    train_path = '../data/erp_fraud/normal_2.csv'
    test_path = '../data/erp_fraud/fraud_3.csv'
    dataset = ERPDataset(train_path=train_path,
                         test_path=test_path,
                         numeric_preprocessing='buckets',
                         categorical_preprocessing='onehot',
                         keep_index=True)
    gold_expl = pd.read_csv('../data/erp_fraud/fraud_3_expls.csv', header=0, index_col=0, encoding='UTF8')
    gold_expl = (gold_expl == 'X').iloc[:, :-5]  # .apply(lambda x: list(x[x.values].index.values), axis=1)
    expls = []
    scores = []
    cos = []
    for i, row in data.iterrows():
        # Explanation values for each feature treated as likelihood of anomalous feature
        #  -aggregated to feature-scores over all feature assignments
        #  -flattened to match shape of y_true
        #  -inverted, so higher score means more anomalous
        cat_expl = xai_to_categorical(expl_df=pd.DataFrame(row).T, dataset=dataset).values.flatten() * -1
        scores.append(roc_auc_score(y_true=gold_expl.loc[i], y_score=cat_expl))
        cos.append(cosine_similarity(gold_expl.loc[i].values.reshape(1, -1), cat_expl.reshape(1, -1))[0, 0])
        expls.append(cat_expl)

    print(f'Mean AUC: {np.mean(scores)}')
    print(f'Std AUC: {np.std(scores)}')
    print(f'Mean Cosine: {np.mean(cos)}')
    print(f'Std Cosine: {np.std(cos)}')
    data = pd.DataFrame(expls, index=data.index)
    heatmap = squareform(pdist(data.apply(lambda row: row > 0.25 * row.max(), axis=1), metric='hamming'))
    # data = gold_expl.loc[fraud_order]
    # heatmap = squareform(pdist(data, metric='hamming'))

    # plot
    font = {'family': 'arial', 'size': 28}
    matplotlib.rc('font', **font)
    matplotlib.rc('text', usetex=True)
    fig, ax = plt.subplots()
    ax = sns.heatmap(heatmap)

    # axis labels
    ticks = [0, 8, 18, 44, 48, 72, 78, 86]
    label_pos_x = [0.15, 0.25, 0.5, 0.1, 0.5, 0.1, 0.3, 0]
    label_pos_y = [-0.175, -0.225, -0.5, -0.1, -0.5, -0.15, -0.25, 0]

    # [Larceny II], [Larceny IV], [Larceny V], [Invoice Kickback I], [Invoice Kickback II], [Corporate Injury]
    plt.xticks(ticks=ticks, rotation=0)
    ax.set_xticklabels(['L1', 'L2', 'L3', 'L4', 'I1', 'I2', 'CI', ''])
    for i, label in enumerate(ax.xaxis.get_majorticklabels()):
        dx = label_pos_x[i]
        offset = matplotlib.transforms.ScaledTranslation(dx, 0, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)
    plt.xlabel('Fraud cases')

    plt.yticks(ticks=ticks, rotation=0)
    ax.set_yticklabels(['L1', 'L2', 'L3', 'L4', 'I1', 'I2', 'CI', ''])
    for i, label in enumerate(ax.yaxis.get_majorticklabels()):
        dy = label_pos_y[i]
        offset = matplotlib.transforms.ScaledTranslation(0, dy, fig.dpi_scale_trans)
        label.set_transform(label.get_transform() + offset)
    plt.ylabel('Fraud cases')

    plt.tight_layout()
    if save:
        # plt.savefig(f'ground_truth_erp.jpg', dpi=400)
        plt.savefig(f'../outputs/explanation/erp_fraud/reproduced/{approach}/{expl_file}.jpg', dpi=400)
    plt.show()
