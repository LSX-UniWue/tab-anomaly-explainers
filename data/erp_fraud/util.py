import os

import numpy as np
import pandas as pd


def get_column_dtypes(source_folder, language):
    """Two lists of column headers for categorical and numerical columns. Needs to be updated on data changes!"""
    column_info = pd.read_csv(os.path.join(source_folder, 'ex2/column_information.csv'), index_col=0,
                              header=0).T
    cat_cols = column_info[column_info['cat'].astype(float) == 1][language].values
    num_cols = column_info[column_info['num'].astype(float) == 1][language].values
    return cat_cols, num_cols


def oversample_data(data, labels):
    """Return dataset that contains equal number of samples for all labels using oversampling"""
    label_counts = labels.value_counts()
    max_counts = label_counts.max()
    sampled_subsets = []
    for label, count in label_counts.items():
        subset = data[labels == label]
        if count < max_counts:  # oversampling
            ids = subset.index.values
            ids = np.concatenate([ids, np.random.choice(subset.index.values, max_counts)], axis=0)
        else:  # use biggest class exactly once
            ids = subset.index.values
        sampled_subsets.append(data.iloc[ids])
    data = pd.concat(sampled_subsets, axis=0).reset_index(drop=True)
    del sampled_subsets
    return data


def adjust_labels_for_task(labels, task_name):
    """Changes some of the labeled anomalies to benign data points depending on task"""
    if task_name == 'session1':
        to_ignore = ['Anomaly_17', 'Anomaly_9']
        labels = labels.replace(to_ignore, 'NonFraud')
    elif task_name == 'session2':
        # Ignore frauds that are only detectable through shipping address (missing in simple BSEG_RSEG join)
        to_ignore = ['Larceny_III', 'Larceny_VI']
        # Ignore frauds that are only detectable through sales information (missing in simple BSEG_RSEG join)
        to_ignore += ['Corporate_Injury_II', 'Selling_Kickback_I_10', 'Selling_Kickback_I_19',
                      'Selling_Kickback_I_20', 'Selling_Kickback_I_30', 'Selling_Kickback_II']
        # Ignore regular rare events that also occur in training data
        to_ignore += ['Scrap', 'Mass_Discount']
        labels = labels.replace(to_ignore, 'NonFraud')

    return labels


def load_erp_dataset(ds_path, column_info, language):
    """
    Loads pandas dataframe from csv and changes language of headers based on header names in column_info
    :param ds_path:             String path to dataset .csv file
    :param column_info_path:    String path to column_info.csv file that includes header translations
    :param language:            String name of language row to use from column_info.csv file
    """
    dataset = pd.read_csv(ds_path, encoding='ISO-8859-1')
    if language not in column_info.columns.values:
        raise ValueError(f'Dataset language needs to be in column_info.csv index.\n'
                         f'Should be one of: {column_info.columns.values} but was: {language}')
    dataset.columns = column_info[language].values.T.reshape(column_info[language].shape[0])
    return dataset


def load_erp_splits(source_folder, dataset_name, language='en'):
    """
    Loads the erp system datasets joined together from BSEG and RSEG tables
    :param source_folder:   String path to data folder
    :param dataset_name:    String name of specific dataset folder
    :param language:        String name of language row to use from column_info.csv file
    """
    if 'session1' in dataset_name:
        folder = 'ex2'
        benign_path = os.path.join(source_folder, folder, 'normal_1.csv')
        fraud_path = os.path.join(source_folder, folder, 'fraud_1.csv')
        column_info = pd.read_csv(os.path.join(source_folder, folder, 'column_information.csv'),
                                  index_col=0,
                                  header=0).T
        benign = load_erp_dataset(ds_path=benign_path, column_info=column_info, language=language)
        fraud = load_erp_dataset(ds_path=fraud_path, column_info=column_info, language=language)

        X_train = benign
        # 50%-50%, making sure to not cut single accounting documents
        X_eval = fraud.iloc[19716:]  # last 50 % as eval data (4 frauds)
        X_test = fraud.iloc[:19716]  # first 50 % as test data (6 frauds)
    elif 'session2' in dataset_name:
        folder = 'ex1'
        benign_path = os.path.join(source_folder, folder, 'normal_2.csv')
        fraud1_path = os.path.join(source_folder, folder, 'fraud_2.csv')
        fraud2_path = os.path.join(source_folder, folder, 'fraud_3.csv')
        column_info = pd.read_csv(os.path.join(source_folder, folder, 'column_information.csv'),
                                  index_col=0,
                                  header=0).T
        benign = load_erp_dataset(ds_path=benign_path, column_info=column_info, language=language)
        fraud1 = load_erp_dataset(ds_path=fraud1_path, column_info=column_info, language=language)
        fraud2 = load_erp_dataset(ds_path=fraud2_path, column_info=column_info, language=language)

        X_train = benign
        X_eval = fraud1
        X_test = fraud2
    else:
        raise ValueError(f'Expected dataset_name string to contain one of '
                         f'["session1", "session2"] but was {dataset_name}')

    return X_train, X_eval, X_test


def preprocessing(data,
                  cat_preprocessor,
                  num_preprocessor,
                  cat_cols,
                  num_cols,
                  fit_new,
                  keep_anomalies=False,
                  keep_original_numeric=False):
    """
    One-hot encoding and standardization for BSEG_RSEG.
    """
    if not keep_anomalies:  # ignore anomalies as they are not detectable from the data used
        data['Label'] = data['Label'].replace({'Anomaly_17': 'NonFraud', 'Anomaly_9': 'NonFraud'})

    # Categorical Preprocessing
    data[cat_cols] = data[cat_cols].astype(str)
    if cat_preprocessor:
        data_cat = data[cat_cols]
        data = data.drop(cat_cols, axis=1)
        if fit_new:
            cat_preprocessor.fit(data_cat)
        data_idx = data_cat.index
        data_cat = cat_preprocessor.transform(data_cat.reset_index(drop=True))
        data_cat.index = data_idx
        data = data.join(data_cat)

    # Numerical Preprocessing
    data_num = data[num_cols]
    if keep_original_numeric:
        data = data.rename({name: name + '_orig' for name in num_cols}, axis=1)
    else:
        data = data.drop(num_cols, axis=1)
    if num_preprocessor:
        if fit_new:
            num_preprocessor.fit(data_num)
        data_idx = data_num.index
        data_num = num_preprocessor.transform(data_num.reset_index(drop=True))
        data_num.index = data_idx
    data = data.join(data_num)

    return data
