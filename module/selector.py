import itertools
import unicodedata
from typing import List

import numpy as np
from sklearn import preprocessing
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler

from featureselection.module.feature_selector import FeatureSelector

from sklearn.utils import class_weight
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
import lightgbm as lgb
import pandas as pd


class FeaturesClear(object):

    def __init__(self, missing_threshold_min=0.20,
                 missing_threshold_max=1,
                 n_iterations: int = 100,
                 correlation_threshold: float = 0.99,
                 cumulative_importance: float = 0.99):
        self.missing_threshold = missing_threshold_min
        self.missing_threshold_max = missing_threshold_max
        self.n_iterations = n_iterations
        self.correlation_threshold = correlation_threshold
        self.cumulative_importance = cumulative_importance

    def preprocessing(self, data, target: str, drop_columns: List, transformation: List,
                      remove_missing_values: bool = False):
        def rcc(s):
            return "".join(ch for ch in s if unicodedata.category(ch)[0] != "C")

        y = data[target]
        for tx in transformation:
            y = y.replace(tx[0], tx[1])
        X = data.drop(columns=[target] + drop_columns)

        X.columns = [
            rcc("".join(c if c.isalnum() else "_" for c in str(x.encode("ascii", errors="ignore").decode()))) for x in
            X.columns]

        X = X.reset_index(drop=True)
        X.replace([np.inf, -np.inf], pd.NaT, inplace=True)

        if remove_missing_values:
            # Missing Values
            print("Missing Values")
            fs = FeatureSelector(data=X, labels=y, random_state=None)
            fs.identify_missing(missing_threshold_min=self.missing_threshold,
                                missing_threshold_max=self.missing_threshold_max)
            missing_features = fs.ops['missing']
            print(missing_features)
            print("removed {0}".format(len(missing_features)))
            self.missing_features = missing_features
            X = X.drop(columns=missing_features)

        object_columns = []
        number_columns = []

        for col in X:
            if (X[col].dtype == object):
                object_columns.append(col)
                X[col] = preprocessing.LabelEncoder().fit_transform(X[col].astype("str").values.reshape(-1, 1))
            else:
                number_columns.append(col)
                X[col] = RobustScaler().fit_transform(X[col].values.reshape(-1, 1))

        return X, y

    def clear(self, X, y, random_state=None):
        # Single Unique Value
        print("Single Unique Value")
        fs = FeatureSelector(data=X, labels=y, random_state=random_state)
        print(fs.identify_single_unique())
        single_unique = fs.ops['single_unique']
        print(single_unique)
        print("removed {0}".format(len(single_unique)))
        self.single_unique = single_unique
        X = X.drop(columns=single_unique)

        # Zero Importance Features
        print("Zero Importance Features")
        fs = FeatureSelector(data=X, labels=y, random_state=random_state)
        fs.identify_zero_importance(task='classification', n_iterations=self.n_iterations, early_stopping=True)
        zero_importance_features = fs.ops['zero_importance']
        print(zero_importance_features)
        print("removed {0}".format(len(zero_importance_features)))
        self.zero_importance_features = zero_importance_features
        X = X.drop(columns=zero_importance_features)

        # Collinear
        print("Collinear")
        fs = FeatureSelector(data=X, labels=y, random_state=random_state)
        fs.identify_collinear(correlation_threshold=self.correlation_threshold)
        collinear = fs.ops['collinear']
        print(collinear)
        print("removed {0}".format(len(collinear)))
        self.collinear = collinear
        X = X.drop(columns=collinear)

        return X, y

    def seletor(self, X, y):
        # Importance Features
        fs = FeatureSelector(data=X, labels=y)
        fs.identify_zero_importance(task='classification', eval_metric='auc',
                                    n_iterations=self.n_iterations, early_stopping=True)
        fs.identify_low_importance(cumulative_importance=self.cumulative_importance)
        low_importance_features = fs.ops['low_importance']
        print(low_importance_features)
        print("removed {0}".format(len(low_importance_features)))
        self.low_importance_features = low_importance_features
        X = X.drop(columns=low_importance_features)
        print("remain {0}".format(len(list(X.columns))))

    def selector_temporal(self, X, y,
                          elements_permu=2,
                          index_column='record_date'
                          , n_estimators: int = 100,
                          learning_rate: float = 0.5,
                          early_stopping_rounds: int = 100,
                          random_state=None):

        scores = []
        importances = []
        importances_mean = np.zeros(len(list(X.columns)) - 1)
        permu = itertools.permutations(X[index_column].unique(), elements_permu)

        cicly = 0
        for index in permu:

            train_index = index[:-1][0]
            test_index = index[-1]

            data_train = X[X[index_column] == train_index]
            target_train = y[X[index_column] == train_index]

            data_test = X[X[index_column] == test_index]
            target_test = y[X[index_column] == test_index]

            data_train = data_train.drop(columns=[index_column])
            data_test = data_test.drop(columns=[index_column])

            class_w_train = list(class_weight.compute_class_weight('balanced', np.unique(target_train), target_train))
            sample_w_train = np.ones(len(target_train), dtype='float')
            for i, val in enumerate(target_train):
                sample_w_train[i] = class_w_train[val]
            lg = lgb.LGBMClassifier(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    metric="custom",
                                    is_unbalance=True, random_state=random_state)
            lg.fit(data_train, target_train, eval_set=(data_test, target_test), verbose=False,

                   sample_weight=sample_w_train, eval_metric=self.lgb_f1_score,
                   early_stopping_rounds=early_stopping_rounds)

            scores.append(lg.best_score_['valid_0']['f1'])
            importances_mean += lg.feature_importances_
            importances.append(lg.feature_importances_)

            cicly += 1

        importances_mean = importances_mean / cicly
        feature_importances = pd.DataFrame(
            {'feature': list(data_train.columns),
             'importance': importances_mean}).sort_values('importance',
                                                          ascending=False).reset_index(drop=True)

        importances = pd.DataFrame(importances, columns=list(data_train.columns))
        importances = importances[list(map(lambda x: x[0], list(feature_importances[["feature"]].values)))]
        agg = importances.agg(["mean", "std"], axis=0).T.reset_index()

        return pd.merge(agg, importances.T.reset_index()), scores

    @staticmethod
    def lgb_f1_score(y_hat, y_true):
        y_true = np.where(y_true < 0.5, 0, 1)
        y_hat = np.where(y_hat < 0.5, 0, 1)
        return 'f1', f1_score(y_true, y_hat), True

    def seletor_kfolds(self, X, y,
                       n_splits: int = 10,
                       n_repeats: int = 5,
                       n_estimators: int = 100,
                       learning_rate: float = 0.5,
                       early_stopping_rounds: int = 100,
                       random_state=None):
        strkfold = RepeatedStratifiedKFold(n_splits=n_splits,
                                           n_repeats=n_repeats,
                                           random_state=random_state)
        kfold_scores = []
        kfold_importances = np.zeros(len(list(X.columns)))
        split_scores = []
        split_importances = np.zeros(len(list(X.columns)))

        for i, (train_indices, valid_indices) in enumerate(strkfold.split(X, y)):
            # Training and validation data
            X_train = X.iloc[train_indices]
            X_valid = X.iloc[valid_indices]
            y_train = y.iloc[train_indices]
            y_valid = y.iloc[valid_indices]

            class_w_train = list(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))
            sample_w_train = np.ones(len(y_train), dtype='float')
            for i, val in enumerate(y_train):
                sample_w_train[i] = class_w_train[val]

            lg = lgb.LGBMClassifier(n_estimators=n_estimators,
                                    learning_rate=learning_rate,
                                    metric="custom",
                                    is_unbalance=True, random_state=None)
            lg.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False,
                   sample_weight=sample_w_train, eval_metric=self.lgb_f1_score,
                   early_stopping_rounds=early_stopping_rounds)

            kfold_scores.append(lg.best_score_['valid_0']['f1'])
            kfold_importances += lg.feature_importances_

        kfold_importances = kfold_importances / (n_repeats * n_splits)

        for i in range(0, (n_repeats * n_splits)):
            X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, stratify=y)
            class_w_train = list(class_weight.compute_class_weight('balanced', np.unique(y_train), y_train))
            sample_w_train = np.ones(len(y_train), dtype='float')
            for i, val in enumerate(y_train):
                sample_w_train[i] = class_w_train[val]

            lg = lgb.LGBMClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                                    metric="custom",
                                    is_unbalance=True, random_state=random_state)
            lg.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=False,
                   sample_weight=sample_w_train, eval_metric=self.lgb_f1_score,
                   early_stopping_rounds=early_stopping_rounds)

            split_scores.append(lg.best_score_['valid_0']['f1'])
            split_importances += lg.feature_importances_

        split_importances = split_importances / (n_repeats * n_splits)

        kfold_importances = kfold_importances / kfold_importances.sum()
        split_importances = split_importances / split_importances.sum()

        mean_importances = (kfold_importances + split_importances) / 2

        feature_importances = pd.DataFrame({'feature': list(X.columns),
                                            'kfold_importance': kfold_importances,
                                            'split_importance': split_importances,
                                            'importance': mean_importances})

        feature_importances = feature_importances.sort_values(by=['kfold_importance'],
                                                              ascending=False).reset_index(drop=True)
        kfold_scores = np.array(kfold_scores)
        split_scores = np.array(split_scores)
        print(
            f'{(n_repeats * n_splits)} KFOLD F1 : {round(kfold_scores.mean(), 5)} with std: {round(kfold_scores.std(), 5)}.')
        print(
            f'{(n_repeats * n_splits)} SPLIT F1 : {round(split_scores.mean(), 5)} with std: {round(split_scores.std(), 5)}.')

        return feature_importances
