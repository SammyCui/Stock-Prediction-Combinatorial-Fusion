import pandas as pd
import numpy as np
from typing import Optional

from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder
from category_encoders.sum_coding import SumEncoder
from category_encoders.m_estimate import MEstimateEncoder
from category_encoders.leave_one_out import LeaveOneOutEncoder
from category_encoders.binary import BinaryEncoder
from category_encoders.cat_boost import CatBoostEncoder
from category_encoders.james_stein import JamesSteinEncoder
from category_encoders.one_hot import OneHotEncoder


def encode_cyclic(df: pd.DataFrame, col: str, n_interval: int,
                  inplace: bool = False, drop_origin: bool = True) -> Optional[pd.DataFrame]:
    """
    encode cyclic data. e.g. month, hour
    :param df:
    :param col:
    :param n_interval: e.g. month: 12, hour: 24
    :param inplace: in-place change
    :param drop_origin: if drop original column
    :return:
    """
    if not inplace:
        return_df = df.copy()
        return_df[col + '_sin'] = np.sin(2 * np.pi * return_df[col] / n_interval)
        return_df[col + '_cos'] = np.cos(2 * np.pi * return_df[col] / n_interval)
        if drop_origin:
            return_df = return_df.drop(columns=[col])
        return return_df
    else:
        df[col + '_sin'] = np.sin(2 * np.pi * df[col] / n_interval)
        df[col + '_cos'] = np.cos(2 * np.pi * df[col] / n_interval)
        if drop_origin:
            df.drop(columns=[col], inplace=True)


def prepare_data(data, L, drop_features, y_feature, test_size, normalize_cols, encoding_method='BinaryEncoder',
                 train_test_split_type='random', normalize_method='RobustScaler',
                 encoding_col='sector', sigma=0, random_seed=40):
    """

  :param data:
  :param L: sequence length
  :param drop_features: features to drop
  :param y_feature:
  :param train_test_split_type: 'random': randomly split ; 'time': split by a quarter e.g., 2019Q4. Default random
  :param test_size: if train_test_slit == 'random': test/all, e.g. 0.3. else: the quarter string to split
  :param encoding_method: str() of encoder to use. Default binaryencoder
  :param encoding_col: col name to encode
  :param sigma: for LOO and catboost encoder -- gaussian noise added to the encoder
  :param random_seed: random_seed for train_test_split under the random mode.
  :return: tuple of X_train, X_test, y_train, y_test
  """

    df = data.copy()
    df = df.sort_values(by=['DATE']).reset_index(drop=True)
    # if split dataset randomly
    if train_test_split_type == 'random':
        X_idx_list = []
        y_idx_list = []
        # TargetEncoder, LOO and CatBoost requires the knowledge of target feature
        # To prevent information leakage, we first store idx for each sequence; then
        # randomly split into train-test, calculate the encodings, and convert back
        for ticker, group in df.groupby('Stock'):
            for i in range(len(group) - L + 1):
                X_idx_list.append(group.iloc[i:i + L].index.to_numpy())
                y_idx_list.append(group.iloc[i + L - 1].name)

        X_train, X_test, y_train, y_test = train_test_split(X_idx_list, y_idx_list, test_size=test_size,
                                                            random_state=random_seed)
    # if split dataset by a time (quarter)
    else:
        train, test = df[df['DATE'] <= test_size].sort_values(by=['DATE']), df[df['DATE'] > test_size].sort_values(
            by=['DATE'])
        train, test = train.reset_index(drop=True), test.reset_index(drop=True)
        X_train, X_test, y_train, y_test = [], [], [], []
        for ticker, group in train.groupby("Stock"):
            for i in range(len(group) - L + 1):
                X_train.append(group.iloc[i:i + L].index.to_numpy())
                y_train.append(group.iloc[i + L - 1].name)
        for ticker, group in test.groupby("Stock"):
            for i in range(len(group) - L + 1):
                X_test.append(group.iloc[i:i + L].index.to_numpy())
                y_test.append(group.iloc[i + L - 1].name)

    # convert back to train df and test df to normalize + category-encoding
    X_train_df_idx, X_test_df_idx = list(set(np.array(X_train).flatten())), list(set(np.array(X_test).flatten()))
    X_train_df, X_test_df = df.loc[X_train_df_idx], df.loc[X_test_df_idx]

    # categorical data encoding

    # ohe and binary expand one cat into multiple binary cols
    if (encoding_method == 'OneHotEncoder') or (encoding_method == 'BinaryEncoder'):
        encoder = eval(encoding_method)()
        encoded_df_train = encoder.fit_transform(X_train_df[encoding_col], X_train_df[y_feature])
        # if it's ohe, drop the last col as it's redundant
        if encoding_method == 'OneHotEncoder':
            encoded_df_train = encoded_df_train.iloc[:, :-1]
        X_train_df = X_train_df.drop(columns=[encoding_col])
        X_train_df = X_train_df.join(encoded_df_train)
        # test set
        encoded_df_test = encoder.transform(X_test_df[encoding_col])
        if encoding_method == 'OneHotEncoder':
            encoded_df_test = encoded_df_test.iloc[:, :-1]
        X_test_df = X_train_df.drop(columns=[encoding_col])
        X_test_df = X_test_df.join(encoded_df_test)
    # target-encoding based methods generate 1 col from 1 cat col
    else:
        if (encoding_method == 'LeaveOneOutEncoder') or (encoding_method == 'CatBoostEncoder'):
            encoder = eval(encoding_method)(sigma=sigma)
        else:
            encoder = eval(encoding_method)()
        X_train_df[encoding_col] = encoder.fit_transform(X_train_df[encoding_col], X_train_df[y_feature])
        X_test_df[encoding_col] = encoder.transform(X_test_df[encoding_col])
        X_train_df, X_test_df = X_train.drop(columns=drop_features), X_test.drop(columns=drop_features)

    if normalize_method is not None:
        normalizer = eval(normalize_method)()
        X_train_df[normalize_cols] = normalizer.fit_transform(X_train_df[normalize_cols])
        X_test_df[normalize_cols] = normalizer.transform(X_test_df[normalize_cols])
    X_train_prepared, X_test_prepared = [], []
    y_train_prepared, y_test_prepared = df.loc[y_train][y_feature], df.loc[y_test][y_feature]
    for sequence in X_train:
        sequence_df = X_train_df.loc[sequence]
        X_train_prepared.append(sequence_df)
    for sequence in X_test:
        sequence_df = X_test_df.loc[sequence]
        X_test_prepared.append(sequence_df)

    return X_train_prepared, X_test_prepared, y_train_prepared, y_test_prepared



from torch.utils.data import Dataset
from torchvision import datasets
drop_features = []
y_feature = ''
