import multiprocessing as mp
from math import ceil
from typing import List

import pandas as pd
import ppscore as pps
from pandas import DataFrame
from toolz import curry
from tsfresh.utilities.dataframe_functions import impute
from varclushi import VarClusHi

from .scorecard import ScorecardSelector
from .statistical_selector import StatisticalSelector
from .utils import corr_matrix, corr_xy, chi, ks
from .feature_selector import FeatureSelector

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
    sc = ScorecardSelector(data,features= features,target= params["target"])
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
    data_dummies = pd.get_dummies(data[features], prefix_sep="_varclus_")
    data_dummies = impute(data_dummies)
    demo1_vc = VarClusHi(data_dummies, maxeigval2=1, maxclus=None)

    # real_columns, categorical_columns, binary_columns = get_feature_type(data, features)
    # data[real_columns + binary_columns] = impute(data[real_columns + binary_columns])
    # demo1_vc = VarClusHi(data[real_columns + binary_columns], maxeigval2=1, maxclus=None)
    clusters = list(map(lambda x: x.clus, list(demo1_vc.varclus().clusters.values())))
    approved = list(set([x.split("_varclus_")[0] for x in sum(n_select(clusters, params.get("percentage", None)), [])]))
    # approved = sum(n_select(clusters, params.get("percentage", None)), []) + categorical_columns
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
