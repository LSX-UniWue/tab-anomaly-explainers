import functools

import tqdm
import numpy as np
import pandas as pd
import sklearn

from data.preprocessing import (CategoricalOneHotPreprocessor,
                                CategoricalOrdinalPreprocessor,
                                NumericalQuantizationPreprocessor)


def gini_index(row, absolute=False):
    # gini index does not hold its meaning for negative values:
    # take absolute or replace negative values with 0s
    if absolute:
        row = np.abs(row)
    else:
        row = np.maximum(row, 0)
    # order non-decreasing
    row = np.sort(row)
    n = len(row)
    l1_norm = np.linalg.norm(row, ord=1)
    if l1_norm == 0:
        # zero vector has uniform distribution of values
        return 0
    # calculate gini index
    gini = 1 - 2 * np.sum((n - np.arange(1, n + 1) + 0.5) * row) / (n * l1_norm)
    return gini


def distraction_weight(expl, gold, absolute=False):
    # test for entire weight or only check feature that speak for an anomaly
    if absolute:
        expl = np.abs(expl)
    else:
        expl = np.maximum(expl, 0)
    # normalize to sum to 1, unless norm is 0
    norm_with_indicative = np.linalg.norm(expl, ord=1)
    if norm_with_indicative == 0:
        return 0
    expl = expl / norm_with_indicative
    # remove weights for correctly assigned relevance
    expl = expl * (1 - gold)
    # get sum of remaining weights: percentage attributed to non-indicative features
    l1_norm = np.linalg.norm(expl, ord=1)
    return l1_norm


def extract_cat_col_mappings():
    """
    Get explanation column names to match columns in original data
    Used in xai_to_categorical()
    """

    def drop_assignments(in_str):
        """Helper fn for removing the _entry appendix for categorical column names"""
        if in_str in ['Betrag_5', 'Kreditkontr_betrag']:  # not categorical, omit
            return in_str
        split = str(in_str).split('_')
        return '_'.join(in_str.split('_')[:-1]) if len(split) > 1 else in_str

    # load data & explanation
    train = pd.read_csv('./data/normal_2.csv', encoding='ISO-8859-1')
    shap_explanation = pd.read_csv('./outputs/shap_eval.csv', index_col=0)
    # X_train, X_eval, X_test, y_eval, y_test = load_splits(source_folder='./data', mode='prep', keep_anomalies=True)

    # remove _assignments from categorical columns
    drop_assignments = np.vectorize(drop_assignments)
    cols = drop_assignments(shap_explanation.columns.values)

    # find unique column names and check if they are the same cols that are in original data
    unique_cols = np.unique(cols, return_counts=True)
    col_dict = {key: val for key, val in zip(unique_cols[0], unique_cols[1])}
    assert not (np.sort(unique_cols[0]) != np.sort(
        train.drop(["Label", "Belegnummer", "Position", "Transaktionsart", "Erfassungsuhrzeit"],
                   axis=1).columns.values)).any()

    # Get list of which columns belong together
    i = 0
    col_names = []
    col_mapping = []
    for col in cols:
        if col in col_dict:
            col_names.append(col)
            col_mapping.append(list(range(i, i + col_dict[col])))
            i += col_dict[col]
            col_dict.pop(col)
    # check that all mappings are correct
    for col_list in col_mapping:
        assert np.unique([cols[x] for x in col_list]).shape[0] == 1

    return col_names, col_mapping


def scores_to_categorical(data, categories):
    """np.concatenate(data_cat, data[:, 29:])
    Slims a data array by adding column values of rows together for all column pairs in list categories.
    Used for summing up scores that were calculated for one-hot representation of categorical features.
    Gives a score for each categorical feature.
    :param data:        np.array of shape (samples, features) with scores from one-hot features
    :param categories:  list with number of features that were used for one-hot encoding each categorical feature
                        (as given by sklearn.OneHotEncoder.categories_)
    :return:
    """
    data_cat = np.zeros((data.shape[0], len(categories)))
    for i, cat in enumerate(categories):
        data_cat[:, i] = np.sum(data[:, cat], axis=1)  # TODO: can max here for non-additive explanations (e.g. Saliency gradient)
    if data.shape[1] > len(categories):  # get all data columns not in categories and append data_cat
        categories_flat = [item for sublist in categories for item in sublist]
        data_cat = np.concatenate((data[:, list(set(range(data.shape[1])) ^ set(categories_flat))], data_cat), axis=1)
    return data_cat


def create_mapping(dataset):
    counter = 0
    mapping_list = []

    num_prep = dataset.preprocessed_data['num_prep']
    cat_prep = dataset.preprocessed_data['cat_prep']

    if isinstance(cat_prep, CategoricalOneHotPreprocessor):
        for cat_mapping in cat_prep.encoder.category_mapping:
            mapping_list.append(list(range(counter, counter + cat_mapping['mapping'].size - 1)))
            counter += cat_mapping['mapping'].size - 1  # -1 because of double nan handling
    elif isinstance(cat_prep, CategoricalOrdinalPreprocessor):
        for _ in dataset.cat_cols:
            mapping_list.append([counter])
            counter += 1
    else:
        raise ValueError(f"Unknown categorical preprocessing: {type(cat_prep).__name__}")

    if isinstance(num_prep, NumericalQuantizationPreprocessor):
        for _ in range(num_prep.encoder.n_bins_.size):
            n_buckets = num_prep.encoder.n_bins + 1
            mapping_list.append(list(range(counter, counter + n_buckets)))
            counter += n_buckets
    else:
        for _ in dataset.num_cols:
            mapping_list.append([counter])
            counter += 1

    return mapping_list


def xai_to_categorical(expl_df, dataset=None):
    """
    Converts XAI scores to categorical values and adds column names

    Example:
    xai_to_categorical(xai_score_path='./scoring/outputs/ERPSim_BSEG_RSEG/pos_shap.csv',
                       out_path='./scoring/outputs/ERPSim_BSEG_RSEG/joint_shap.csv',
                       data_path='./datasets/real/ERPSim/BSEG_RSEG/ERP_Fraud_PJS1920_BSEG_RSEG.csv')
    """
    cat_cols = create_mapping(dataset)

    col_names = dataset.get_column_names()

    expls_joint = scores_to_categorical(expl_df.values, cat_cols)

    return pd.DataFrame(expls_joint, index=expl_df.index, columns=col_names)


def get_benign_knns(X_anom, X_benign, booking_key_only=False, k=5):
    """
    Create kNN for each transaction key in X_anom and find k neighbors
    Takes neighbors either from all X_benign data, or from data where
    X_anom['Buchungsschuessel'] == X_benign['Buchungsschuessel']
    """
    if booking_key_only:
        # filter all keys occuring in X_anom
        keys = X_anom.loc[:, X_anom.columns.str.startswith('Buchungsschluessel')]
        keys = keys.columns[(keys == 1).any()].values

        # get neighbors
        all_neighbors = []
        for key in keys:
            data_benign = X_benign[X_benign[key] == 1]
            data_anom = X_anom[X_anom[key] == 1]
            knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k).fit(X=data_benign,
                                                                            y=np.ones(data_benign.shape[0]))
            neighbors = pd.DataFrame(knn.kneighbors(X=data_anom, return_distance=False), index=data_anom.index)
            neighbors = neighbors.apply(lambda x: pd.Series(data_benign.iloc[x].index.values),
                                        axis=1)  # original benign idxs
            all_neighbors.append(neighbors)
        return pd.concat(all_neighbors)
    else:
        # get neighbors
        all_neighbors = []
        knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors=k).fit(X=X_benign, y=np.ones(X_benign.shape[0]))
        neighbors = pd.DataFrame(knn.kneighbors(X=X_anom, return_distance=False), index=X_anom.index)
        neighbors = neighbors.apply(lambda x: pd.Series(X_benign.iloc[x].index.values), axis=1)  # original benign idxs
        return neighbors


def image_reference_points(background, X_expl, mvtec_data=None, predict_fn=None, device=None):
    if background == 'zeros':  # zero vector, default
        reference_points = np.zeros(X_expl.shape)

    elif background == 'random_benign':  # a benign sample to use as baseline
        if mvtec_data is None:
            raise ValueError("background 'random_benign' requires mvtec data object as input at variable 'mvtec_data'")
        reference_points = next(iter(mvtec_data.get_data_loader(dataset_name='train')))[0][1].resize(1, *X_expl[0].shape)  # TODO: shuffle + repeat

    elif background == 'NN':  # nearest neighbor in the normal training data
        if mvtec_data is None:
            raise ValueError("background 'NN' requires mvtec data object as input at variable 'mvtec_data'")
        from sklearn.neighbors import NearestNeighbors
        X_train, _, _ = mvtec_data.get_full_dataset('train')
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_train.reshape(X_train.shape[0], -1))
        neighbor_dist, neighbor_idx = nbrs.kneighbors(X=X_expl.reshape(X_expl.shape[0], -1), n_neighbors=1, return_distance=True)
        reference_points = X_train[neighbor_idx.flatten()]

    elif background == 'mean':  # mean training data point
        if mvtec_data is None:
            raise ValueError("background 'mean' requires mvtec data object as input at variable 'mvtec_data'")
        X_train, _, _ = mvtec_data.get_full_dataset('train')
        reference_points = np.mean(X_train, axis=0).reshape((1, *X_train.shape[1:])).repeat(X_expl.shape[0], axis=0)

    elif background == 'optimized':  # optimized input in vicinity of the anomaly that the network predicts as benign
        if predict_fn is None or device is None:
            raise ValueError("background 'optimized' requires predict_fn and device as input")
        from xai.automated_background_torch import optimize_input_gradient_descent
        reference_points = np.zeros(X_expl.shape)
        for i in range(X_expl.shape[0]):
            reference_points[i] = optimize_input_gradient_descent(data_point=X_expl[i].reshape((1, *X_expl.shape[1:])),
                                                                  mask=np.zeros((1, *X_expl.shape[1:])),
                                                                  predict_fn=functools.partial(predict_fn, output_cpu=False),
                                                                  device=device)

    else:
        reference_points = None

    return reference_points


def tabular_reference_points(background, X_expl, X_train=None, columns=None, predict_fn=None):

    if background in ['mean', 'NN']:
        assert X_train is not None, f"background '{background}' requires train data as input at variable 'X_train'"
    if background in ['optimized']:
        assert predict_fn is not None, f"background '{background}' requires predict_fn as input"

    if background == 'zeros':  # zero vector, default
        reference_points = np.zeros(X_expl.shape)
        return reference_points

    elif background == 'mean':  # mean training data point for each data point
        reference_points = np.mean(X_train, axis=0).reshape((1, -1)).repeat(X_expl.shape[0], axis=0)
        return reference_points

    elif background == 'NN':  # nearest neighbor in the normal training data for each data point
        from sklearn.neighbors import NearestNeighbors
        nbrs = NearestNeighbors(n_neighbors=1).fit(X_train)
        neighbor_dist, neighbor_idx = nbrs.kneighbors(X=X_expl, n_neighbors=1, return_distance=True)
        reference_points = X_train[neighbor_idx.flatten()]
        return reference_points

    elif background == 'single_optimum':  # one normal point in the proximity for each data point
        from xai.automated_background_torch import optimize_input_quasi_newton
        reference_points = np.zeros(X_expl.shape)
        for i in tqdm.tqdm(range(X_expl.shape[0]), desc='generating reference points'):
            reference_points[i] = optimize_input_quasi_newton(data_point=X_expl[i].reshape((1, -1)),
                                                              kept_feature_idx=None,
                                                              predict_fn=predict_fn)
        return reference_points

    elif background == 'kmeans':  # kmeans cluster centers of normal data as global background
        from sklearn.cluster import k_means
        centers, _, _ = k_means(X=X_train, n_clusters=5)
        return centers

    else:
        raise ValueError(f"Unknown background: {background}")
