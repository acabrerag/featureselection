import gc
import multiprocessing as mp
from itertools import combinations
from typing import List, Optional
import dask.dataframe as dd
import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, roc_auc_score



def read_csv_batch(name, **kwargs) -> pd.DataFrame:
    return pd.concat([x for x in pd.read_csv("{0}".format(name),
                                             chunksize=kwargs.pop("chunksize", 500),
                                             low_memory=kwargs.pop("low_memory", False), **kwargs)])


def read_csv_batch_dask(name, **kwargs) -> pd.DataFrame:
    """
    read a dataframe in row batches. . This function needs lower RAM than pd.read_csv
    Parameters
    ----------
    name
    kwargs
    Returns
    -------
    """
    chunksize = kwargs.pop("chunksize", 10000)
    low_memory = kwargs.pop("low_memory", False)

    chunk_list = []
    df_test = pd.read_csv("{0}".format(name), chunksize=chunksize, low_memory=low_memory, **kwargs)
    for chunk in df_test:
        chunk_list.append(dd.from_pandas(chunk, npartitions=10))
    del df_test
    gc.collect()
    return dd.multi.concat(chunk_list)


def SelectKBest(result: dict, types_features="numerical", type_target="bad", k: int = 5):
    features = list(result[type_target][types_features].keys())
    return features[0:min(k, len(features))]


def roc_weighted(y_test, y_pred):
    """
    compute the roc score
    y_test: actual label
    y_pred:probabitily prediction

    """
    return roc_auc_score(y_test, y_pred, average='weighted')


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
            df[target]=df[target].astype(int)
            raise Exception("mixed types")
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


def import_class_rec(dictionary):
    def creator(sign: str, params: Optional[dict]):
        if sign is not None and isinstance(params, dict):
            module_name, _, class_name = sign.rpartition('.')
            try:
                return getattr(
                    __import__(module_name, globals(), locals(), [class_name], 0),
                    class_name)(**params)
            except Exception as e:
                pass
        return sign if bool(params) is False else {sign: params} if sign is not None else params

    if isinstance(dictionary, dict):
        for k, v in dictionary.items():
            return creator("featureselection.module.functions." + k, v)
