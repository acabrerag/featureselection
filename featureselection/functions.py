import multiprocessing as mp
from math import ceil

import pandas as pd
import ppscore as pps
from pandas import DataFrame
from toolz import curry
from tsfresh.utilities.dataframe_functions import impute
from varclushi import VarClusHi

from .scorecard import ScorecardSelector
from .statistical_selector import StatisticalSelector
from .feature_selector import FeatureSelector
from itertools import combinations
from typing import List
import numpy as np
import scipy.stats as stats
from sklearn.metrics import confusion_matrix

pd.options.mode.chained_assignment = None


@curry
def get_feature_type(data: DataFrame, features: List):
    categorical_columns = []
    real_columns = []
    binary_columns = []
    for col in features:
        if (data[col].dtype == object) | (data[col].dtype == bool):
            n_unique_values = len(set(data[col].values))
            if n_unique_values == 2 and 1 in set(data[col].values) or True in set(data[col].values):
                binary_columns.append(col)
            else:
                categorical_columns.append(col)
        else:
            real_columns.append(col)

    return real_columns, categorical_columns, binary_columns


@curry
def clear_filter(data: DataFrame, features: List, params: dict):
    fs = FeatureSelector(data=data[features], random_state=0)
    fs.identify_single_unique()
    fs.remove(methods=["single_unique"], keep_one_hot=False)
    fs.identify_missing(missing_threshold_min=-0.1,
                        missing_threshold_max=params.get("missing_threshold_max", 0.8))
    data = fs.remove(methods=['missing'], keep_one_hot=False)
    data = data.T.drop_duplicates().T
    approved = data.columns.to_list()
    not_approved = list(set(features) - set(approved))

    return data, approved, not_approved, {"clear_filter": {"approved": approved,
                                                           "not_approved": not_approved}
                                          }


@curry
def statistical_test(data: DataFrame, features: List, params: dict):
    target = params["target"]
    n_jobs = params.get("n_jobs", 1)
    real_columns, categorical_columns, binary_columns = get_feature_type(data, features)
    data[real_columns] = impute(data[real_columns])
    df_0 = data[data[target] == 0]
    df_1 = data[data[target] == 1]
    p_values = {}
    pool = mp.Pool(n_jobs)
    iterat = [[df_0[feat], df_1[feat]] for feat in features]
    results = pool.starmap(ks, iterat)
    pool.close()
    p_values = dict(zip(features, results))
    approved_numerical = list(filter(lambda x: p_values[x] < params.get("alpha", 0.05), p_values.keys()))

    pool = mp.Pool(n_jobs)
    iterat = [[pd.crosstab(data[feat], data[target])] for feat in categorical_columns]
    results = pool.starmap(chi, iterat)
    pool.close()
    p_values = dict(zip(features, results))
    approved_categorical = list(filter(lambda x: p_values[x] < params.get("alpha", 0.05), p_values.keys()))
    approved = approved_categorical + approved_numerical
    not_approved = list(set(features) - set(approved))
    return data, approved, not_approved, {
        "statistical_test": {"approved": approved,
                             "not_approved": not_approved}
    }


@curry
def information_value_filter(data: DataFrame, features: List, params: dict):
    sc = ScorecardSelector(data, features=features, target=params["target"])
    approved = sc.filter(elements=features, iv_limit=0.25, missing_limit=1, identical_limit=1)
    not_approved = list(set(features) - set(approved))

    return data, approved, not_approved, {
        "information_value_filter": {"approved": approved,
                                     "not_approved": not_approved}
    }


@curry
def pps_filter(data: DataFrame, features: List, params: dict):
    n_jobs = params.get("n_jobs", 8)
    data[params["target"]] = data[params["target"]].astype(str)

    iterat = [[data[[feat1] + [params["target"]]], feat1, params["target"]] for feat1 in
              features]
    pool = mp.Pool(n_jobs)
    results = pool.starmap(pps.score, iterat)
    pool.close()

    results = sorted(results, key=lambda i: i['ppscore'], reverse=True)
    DataFrame(results).to_csv(params.get("output_dir", "../data/") + "pps.csv", index=False)
    approved = list(map(lambda x: x['x'], filter(lambda x: x['ppscore'] < params.get("threshold", 0.01), results)))
    not_approved = list(set(features) - set(approved))

    return data, approved, not_approved, {
        "pps_filter": {"approved": approved,
                       "not_approved": not_approved}
    }


@curry
def n_select(clusters, percentage):
    if 0 < percentage < 1:
        return list(map(lambda x: x[:ceil(len(x) * percentage)], clusters))
    else:
        return list(map(lambda x: x[:ceil(percentage)], clusters))


@curry
def varclus_filter(data: DataFrame, features: List, params: dict):
    demo1_vc = VarClusHi(impute(pd.get_dummies(data[features], prefix_sep="_varclus_")), maxeigval2=1, maxclus=None)
    clusters = list(map(lambda x: x.clus, list(demo1_vc.varclus().clusters.values())))
    approved = list(set([x.split("_varclus_")[0] for x in sum(n_select(clusters, params.get("percentage", None)), [])]))
    not_approved = list(set(features) - set(approved))
    return data, approved, not_approved, {
        "varclus_filter": {"approved": approved,
                           "not_approved": not_approved}
    }


@curry
def correlation_filter(data: DataFrame, features: List, params: dict):
    n_jobs = params.get("n_jobs", 1)
    target = params["target"]
    coef_numerical = params.get("coef_numerical", 0.6)
    percentage_numerical = params.get("percentage_numerical", 0.5)
    coef_categorical = params.get("coef_categorical", 0.2)
    percentage_categorical = params.get("percentage_categorical", 0.5)
    real_columns, categorical_columns, binary_columns = get_feature_type(data, features)

    def aux(data, columns, target, types_columns, coef, percentage, n_jobs):
        if len(columns) > 1:
            corxy_dict = corr_xy(data, columns, target, types_columns=types_columns, n_jobs=n_jobs)
            order_list = [x[0] for x in sorted(corxy_dict.items(), key=lambda x: x[1], reverse=True)]
            cor_matrix = corr_matrix(data, columns, types_columns, n_jobs=n_jobs)
            sc = StatisticalSelector(data, target).correlation_selector(corr_matrix=cor_matrix,
                                                                        order_list=order_list,
                                                                        coef=coef,
                                                                        percentage=percentage)
            return sc["selected_features"]
        else:
            return columns

    approved_categorical = aux(data, categorical_columns + binary_columns, target, "categorical", coef_categorical,
                               percentage_categorical, n_jobs)
    approved_numerical = aux(data, real_columns, target, "numerical", coef_numerical, percentage_numerical, n_jobs)

    approved = approved_numerical + approved_categorical
    not_approved = list(set(features) - set(approved))
    return data, approved, not_approved, {
        "correlation_filter": {"approved": approved,
                               "not_approved": not_approved}
    }


@curry
def feature_importance_filter(data: DataFrame, features: List, params: dict):
    fs = FeatureSelector(data=data[features],
                         labels=data[params["target"]],
                         random_state=params.get("random_state", None))
    fs.identify_zero_importance(task="classification",
                                n_iterations=params.get("n_iterations", 5),
                                early_stopping=params.get("early_stopping", True),
                                algorithm=params.get("algorithm", "LightGBM"))
    fs.identify_low_importance(cumulative_importance=params.get("cumulative_importance", 0.95))
    fs.remove(methods=['zero_importance', 'low_importance'], keep_one_hot=False)
    not_approved = fs.removed_features
    approved = list(set(features) - set(not_approved))

    return data, approved, not_approved, {
        "feature_importance_filter": {"approved": approved,
                                      "not_approved": not_approved}
    }


def corr_matrix(df: pd.DataFrame, features: List[str], types_columns: str = "numerical", n_jobs: int = 1, **kwargs):
    if types_columns == "numerical":
        corr_matrix = df[features].corr()
    else:
        nan_encoder = kwargs.get("nan_encoder", True)
        comb = list(combinations(features, 2))
        iterat = [[df[feat1], df[feat2], nan_encoder] for feat1, feat2 in comb]

        pool = mp.Pool(n_jobs)
        results = pool.starmap(cramers_corrected_stat, iterat)
        pool.close()
        corr_matrix = pd.DataFrame(columns=features, index=features)
        for i, (col1, col2) in enumerate(comb):
            corr_matrix[col1][col2] = results[i]
            corr_matrix[col2][col1] = corr_matrix[col1][col2]
            corr_matrix[col2][col2] = 1
    return corr_matrix


def pearson(x, y):
    return np.corrcoef(x, y)[0][1]


def corr_xy(df: pd.DataFrame, features: List[str], target: str, types_columns: str, n_jobs: int = 1):
    """
    function to compute the correlation between features and a target
    Use Pearson correlation for numeric features: target need to be numeric
    Use Cramers phi for categorical features: target need to be categorical
    """

    if types_columns == "numerical":
        if df[target].dtype == object:
            df[target] = df[target].astype(int)
        iterat = [[df[feat], df[target]] for feat in features]
        pool = mp.Pool(n_jobs)
        results = pool.starmap(pearson, iterat)
    else:
        iterat = [[df[feat], df[target]] for feat in features]
        pool = mp.Pool(n_jobs)
        results = pool.starmap(cramers_corrected_stat, iterat)
    pool.close()
    corxy_dict = dict(zip(features, results))
    return corxy_dict


def ks(x, y):
    return stats.ks_2samp(x, y)[1]


def chi(conf_matrix):
    return stats.chi2_contingency(conf_matrix, correction=conf_matrix.shape[0] != 2)[1]


def check_dtype(df, features):
    features_type = {"numerical": set([col for col in features if df[col].dtype not in ['object']])}
    features_type.update({"categorical": set(features) - features_type["numerical"]})
    features_type.update({"nan": set([col for col in features_type["numerical"] if len(df) == df[col].isna().sum()])})

    return features_type


def outlier_treatment(datacolumn: pd.core.series.Series, quantil=[25, 75], multiplier=1.5) -> tuple:
    Q1, Q3 = np.nanpercentile(datacolumn, quantil)
    IQR = Q3 - Q1
    lower_range = Q1 - (multiplier * IQR)
    upper_range = Q3 + (multiplier * IQR)
    return lower_range, upper_range


def get_numeric(df: pd.DataFrame) -> tuple:
    object_columns = []
    number_columns = []

    for col in df:
        if (df[col].dtype == object) | (df[col].dtype == bool):
            object_columns.append(col)
        else:
            number_columns.append(col)
    return number_columns, object_columns


def get_caps_floors(df: pd.DataFrame, quantil=[25, 75], multiplier=1.5) -> tuple:
    number_columns, object_columns = get_numeric(df)
    no_floor_cap_var = [col for col in number_columns if df[col].median() == 0]
    floor_cap_var = list(set(number_columns) - set(no_floor_cap_var))
    precomputed_caps = {}
    precomputed_floors = {}
    for column in floor_cap_var:
        l, u = outlier_treatment(df[column], quantil, multiplier)
        precomputed_floors[column] = float(l)
        precomputed_caps[column] = float(u)

    return precomputed_caps, precomputed_floors


def SelectKBest(result: dict, types_features="numerical", type_target="bad", k: int = 5):
    features = list(result[type_target][types_features].keys())
    return features[0:min(k, len(features))]


def weighted(y_test, y_pred, wb=0.7):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = np.array(cm).ravel()
    good = (1 / (tn + fp)) * tn
    bad = (1 / (fn + tp)) * tp
    return (1 - wb) * good + wb * bad


def good(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = np.array(cm).ravel()
    return (1 / (tn + fp)) * tn


def bad(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = np.array(cm).ravel()
    return (1 / (fn + tp)) * tp


def list_to_file(list: List, path: str):
    import pickle
    with open(path, 'wb') as filehandle:
        pickle.dump(list, filehandle)


def cramers_corrected_stat(x: pd.Series, y: pd.Series, nan_encoder: bool = True):
    """ calculate Cramers V statistic for categorial-categorial association.
        uses correction from Bergsma and Wicher,
        Journal of the Korean Statistical Society 42 (2013): 323-328
    """

    """
    For nan we will create another category in the fist     
    """
    if nan_encoder:
        x = x.fillna("cramers_corrected_stat_nan")
        y = y.fillna("cramers_corrected_stat_nan")
    result = np.nan
    if len(x.value_counts()) == 1:
        print("First variable is constant")
    elif len(y.value_counts()) == 1:
        print("Second variable is constant")
    else:
        conf_matrix = pd.crosstab(x, y)
        if conf_matrix.shape != (0, 0):

            if conf_matrix.shape[0] == 2:
                correct = False
            else:
                correct = True

            chi2 = stats.chi2_contingency(conf_matrix, correction=correct)[0]
            n = sum(conf_matrix.sum())
            if n == 1:
                return 0
            phi2 = chi2 / n
            r, k = conf_matrix.shape
            phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
            if phi2corr == 0:
                return 0
            rcorr = r - ((r - 1) ** 2) / (n - 1)
            kcorr = k - ((k - 1) ** 2) / (n - 1)
            if min((kcorr - 1), (rcorr - 1)) == 0:
                return 0
            result = np.sqrt(phi2corr / min((kcorr - 1), (rcorr - 1)))
    return round(result, 6)
