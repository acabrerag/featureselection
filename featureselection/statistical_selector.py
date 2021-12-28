import multiprocessing as mp
from datetime import date
from itertools import chain, combinations
from typing import List
import scipy.stats as stats

import numpy as np
import pandas as pd
from fklearn.training.transformation import onehot_categorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split


def roc_weighted(y_test, y_pred):
    """
    compute the roc score
    y_test: actual label
    y_pred:probabitily prediction

    """
    return roc_auc_score(y_test, y_pred, average='weighted')


class StatisticalSelector(object):
    """
        Class for performing feature selection for binary classification problems.
        Implements two different methods to identify features for removal
            1. Find columns with different distribution for target variable
            2. Find columns with good performance to identify TP, TN and best
        Parameters
        --------
            data : dataframe
                A dataset with observations in the rows and features in the columns
            target : string
                Name of the column with the target for classification
    """

    def __init__(self, data: pd.DataFrame, target: str, random_state=None):

        self.data = data
        self.target = target
        self.random_state = random_state
        self.tracker = new_tracker(["feature", "type", "step", "removed", "step_2", "step_5",
                                    "step_6", "step_2_threshold", "step_5_threshold", "step_6_threshold"])

    def preproc(self, features: List[str]):
        """
        Function for data preprocessing
        Parameters
        ----------
        features: List
            List of features names in the dataset
        Returns
            preproccess data
            list of numeric features
            list of categorial features
        -------
        """
        from fklearn.training.transformation import floorer, capper
        from fklearn.training.pipeline import build_pipeline
        from fklearn.training.imputation import imputer
        self.data = self.data.replace([np.inf, -np.inf], np.nan)
        number_columns, object_columns = get_numeric(self.data[features])
        if number_columns:
            caps, floors = get_caps_floors(self.data)
            floor = floorer(columns_to_floor=list(floors.keys()), precomputed_floors=floors)
            cap = capper(columns_to_cap=list(caps.keys()), precomputed_caps=caps)
            imputer_number = imputer(columns_to_impute=number_columns, impute_strategy='median')
            pipeline = build_pipeline(floor, cap, imputer_number)
            _, self.data, __ = pipeline(self.data)
        return number_columns, object_columns

    def stat_selector(self, features: List[str], types_columns: str = "numerical", alpha: float = 0.05,
                      n_jobs: int = 1):
        """
        Function to select features based on the Kolmogorov-Smirnov test (numerical) and the chi-test (categorical)
        Parameters
        ----------
        features: List of features
        types_columns: numerical or categorical
        alpha: significance level of the test
        n_jobs: number of cores to use
        Returns
        -------
            list of selected features
        """
        types = check_dtype(self.data, features)
        # if types[types_columns] != set(features):
        #     if not (types_columns == "categorical") & (types["nan"] == types["numerical"]):
        #         raise Exception("mixed types")
        df_0 = self.data[self.data[self.target] == 0]
        df_1 = self.data[self.data[self.target] == 1]
        p_values = {}
        pool = mp.Pool(n_jobs)
        if types_columns == "numerical":
            iterat = [[df_0[feat], df_1[feat]] for feat in features]
            results = pool.starmap(ks, iterat)
        elif types_columns == "categorical":
            iterat = [[pd.crosstab(self.data[feat], self.data[self.target])] for feat in features]
            results = pool.starmap(chi, iterat)
        else:
            raise NotImplementedError("{0} not sopported".format(types_columns))
        pool.close()
        p_values = dict(zip(features, results))
        selected_features = list(filter(lambda x: p_values[x] < alpha, p_values.keys()))
        return {"selected_features": selected_features,
                "removed_features": list(set(features) - set(selected_features)), "p_values": p_values, "alpha": alpha}

    def correlation_selector(self, corr_matrix: pd.DataFrame, order_list: List[str], coef: float = 0.6,
                             percentage: float = 1):
        """
        Function to select features based on correlation groups
        Parameters
        ----------
        corr_matrix: DataFrame with correlations
        order_list: Priority order to select the groups
        coef: minimum correlation value for group the features
        percentage: float to indicate the percentage of the features to be selected in each group,
                    if is int indicate the number of features to be selected in each group.
        Returns
        -------
            list of selected features
        """
        corr_matrix = abs(corr_matrix)
        already_in = set()
        groups_corr = []
        for col in order_list:
            perfect_corr = corr_matrix[col][corr_matrix[col] >= coef].index.tolist()
            perfect_corr = list(set(perfect_corr) - already_in)
            if perfect_corr and col not in already_in:
                already_in.update(set(perfect_corr))
                groups_corr.append(perfect_corr)
        selected_features = []
        for group in groups_corr:
            if percentage < 1:
                n = int(percentage * len(group))
                n = n if n >= 1 else 1
            else:
                n = percentage
            selected_features.append(group[0:n])
        selected_features = list(chain(*selected_features))
        return {"selected_features": selected_features, "groups": groups_corr}

    def cm_selector(self, features: List[str], types_columns="numerical", lower: float = 0.5, lower_best=0.5,
                    n_jobs: int = 1, best_function=roc_weighted, test_size=0.3, LRCV: dict = {}):
        """
        Function to select features based on confusion matrix result for a logistic regression
        Parameters
        ----------
        features: list of features
        types_columns: numerical or categorical
        lower: minimum value good/bad rate
        lower_best: minumum value for best-metric
        n_jobs: number of core to use
        best_function: function to evaluate in the best feature selection
                        the default is a weighted roc.
        Returns
        -------
            dict with selected features for good, bad rates and best features
        """
        best, good, bad = [], [], []
        y = self.data[self.target]
        cv = LRCV.get("cv", 10)
        max_iter = LRCV.get("max_iter", 5000)
        solver = LRCV.get("solver", 'liblinear')
        class_weight = LRCV.get("class_weight", "balanced")

        for feat in features:
            X = self.data[[feat]]
            if types_columns == "categorical":
                onehot = onehot_categorizer(columns_to_categorize=[feat], hardcode_nans=True, drop_first_column=True)
                a, X, _ = onehot(X)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size,
                                                                random_state=self.random_state,
                                                                stratify=y)
            logreg = LogisticRegressionCV(cv=cv, max_iter=max_iter, solver=solver, random_state=self.random_state,
                                          class_weight=class_weight, n_jobs=n_jobs)
            logreg.fit(X_train, y_train)
            y_pred = logreg.predict(X_test)
            best.append(best_function(y_test, y_pred))
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = np.array(cm).ravel()
            good.append((1 / (tn + fp)) * tn)
            bad.append((1 / (fn + tp)) * tp)
        best_dict = dict(zip(features, best))
        good_dict = dict(zip(features, good))
        bad_dict = dict(zip(features, bad))
        good_list = [k for k, v in sorted(good_dict.items(), key=lambda item: item[1], reverse=True) if v >= lower]
        bad_list = [k for k, v in sorted(bad_dict.items(), key=lambda item: item[1], reverse=True) if v >= lower]
        best_list = [k for k, v in sorted(best_dict.items(), key=lambda item: item[1], reverse=True)
                     if v >= lower_best]
        return {"selected": {"best": best_list, "good": good_list, "bad": bad_list},
                "all": {"best": best_dict, "bad": bad_dict, "good": good_dict},
                "thresholds": {"lower_best": lower_best, "lower": lower}}

    def select_features_by_type(self, features: List[str], types_columns: str = "numerical", n_jobs: int = 1,
                                **kwargs):
        """
        function to select features by type
        Parameters
        ----------
        features: List of features
        types_columns: numerical or categorical
        n_jobs: number of cores
        Returns
            dict with:
                features for detection of negatives:0 (goods)
                features for detection of positives:1 (bads)
                best features
        """
        print("Step 5 for {0} features".format(types_columns))
        # Step x1: Remove by stat selector
        old_selection = features
        if kwargs.get("stat_selector", None) is not None:
            res = self.stat_selector(features, types_columns, n_jobs=n_jobs,
                                     **kwargs.get("stat_selector", {}))
            selected_features = res["selected_features"]
            removed_features = list(set(old_selection) - set(selected_features))
            update_tracker(self.tracker, feature=old_selection, type=types_columns,
                           step=5, removed=[1 if k in removed_features else 0 for k in old_selection],
                           step_5=[res["p_values"][k] for k in old_selection], step_5_threshold=res["alpha"])
        else:
            selected_features = features
            removed_features = []
            print("stat_selector not used")

        print("Step 6 for {0} features".format(types_columns))

        if kwargs.get("cm_selector", None) is not None:
            # Step x2: Remove by confusion matrix selector
            res = self.cm_selector(selected_features, types_columns, **kwargs.get("cm_selector", {}))
            res_dict = res["all"]
            best_dict = res_dict["best"]
            old_selection = selected_features
            selected_features = res["selected"]["best"]
            update_tracker(self.tracker, feature=old_selection, type=types_columns,
                           step=6, removed=[0 if k in selected_features else 1 for k in old_selection],
                           step_6=[best_dict[k] for k in old_selection],
                           step_6_threshold=res["thresholds"]["lower_best"])
        else:
            selected_features = features
            removed_features = []
            print("cm_selector not used")

        print("Step 7 for {0} features".format(types_columns))

        if kwargs.get("correlation_selector", None) is not None:
            # Step x3: Remove by correlation selector as a bad detection feature
            old_selection = selected_features
            bad_feats = res["selected"]["bad"]
            corxy_dict = corr_xy(self.data, bad_feats, self.target, types_columns=types_columns, n_jobs=n_jobs)
            corxy_dict = {k: abs(v) for k, v in corxy_dict.items()}
            order_list = [x[0] for x in sorted(corxy_dict.items(), key=lambda x: x[1], reverse=True)]
            cor_matrix = corr_matrix(self.data, order_list, types_columns, n_jobs=n_jobs)
            res_co = self.correlation_selector(cor_matrix, order_list,
                                               **kwargs.get("correlation_selector", {}))
            selected_features = res_co["selected_features"]
            bad_feats = list(set(bad_feats) & set(selected_features))
            # step x3: Remove by correlation selector as a good detection feature
            good_feats = res["selected"]["good"]
            corxy_dict = corr_xy(self.data, good_feats, self.target, types_columns=types_columns)
            corxy_dict = {k: abs(v) for k, v in corxy_dict.items()}
            order_list = [x[0] for x in sorted(corxy_dict.items(), key=lambda x: x[1], reverse=True)]
            cor_matrix = corr_matrix(self.data, order_list, types_columns, n_jobs=n_jobs)
            res_co = self.correlation_selector(cor_matrix, order_list,
                                               **kwargs.get("correlation_selector", {}))
            selected_features = res_co["selected_features"]

            good_feats = list(set(good_feats) & set(selected_features))
            # step x3: Remove by correlation selector as a best feature
            order_list = [k for k, v in sorted(best_dict.items(), key=lambda item: item[1], reverse=True)
                          if k in old_selection]
            cor_matrix = corr_matrix(self.data, order_list, types_columns, n_jobs=n_jobs)
            res_co = self.correlation_selector(cor_matrix, order_list,
                                               **kwargs.get("correlation_selector", {}))
            selected_features = res_co["selected_features"]
            best_feats = list(set(res["selected"]["best"]) & set(selected_features))
            selected_features = best_feats
            update_tracker(self.tracker, feature=old_selection, type=types_columns,
                           step=7, removed=[0 if k in selected_features else 1 for k in old_selection])
        else:
            good_feats = features
            bad_feats = features
            best_feats = features
            res_dict = {}
            print("correlation_selector not used")

        return {"selected": {"good": good_feats, "bad": bad_feats, "best": best_feats}, "all": res_dict}

    def select_features(self, features: List[str], n_jobs: int = 1, **kwargs):
        """
        coef=0.25 for object data is suggested for a very strong association
        and coef=0.15 for a strong association
        https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6107969/
        Parameters
        ----------
        features
        Returns
            dict with:
                features (numerical and categorical) for detection of negatives:0 (goods)
                features (numerical and categorical) for detection of positives:1 (bads)
                dict of final features (numerical and categorical):
        -------
        """
        number_columns, object_columns = self.preproc(features)

        print("Selecting categorical features")
        if object_columns:
            res = self.select_features_by_type(object_columns, types_columns="categorical", n_jobs=n_jobs,
                                               **kwargs.get("categorical", {}))
            res = res["selected"]

        else:
            print("There are not categorical features")
            res = {"good": [], "bad": [], "best": []}
        good_ob, bad_ob, best_ob = res['good'], res['bad'], res['best']

        print("Selecting numerical features")
        if number_columns:

            res = self.select_features_by_type(number_columns, types_columns="numerical", n_jobs=n_jobs,
                                               **kwargs.get("numerical", {}))
            res = res["selected"]
        else:
            print("There are not numerical features")
            res = {"good": [], "bad": [], "best": []}

        good, bad, best = res['good'], res['bad'], res['best']

        return {"selected": {"good": good + good_ob, "bad": bad + bad_ob, "best": best + best_ob},
                "all": {"good": {"numerical": good, "categorical": good_ob},
                        "bad": {"numerical": bad, "categorical": bad_ob},
                        "best": {"numerical": best, "categorical": best_ob}}}


def new_tracker(fields):
    return {f: [] for f in fields}


def update_tracker(tracker, feature, **kwargs):
    n = len(feature)
    missing_fields = set(list(tracker.keys())) - set(list(kwargs.keys()) + ["feature"])

    def aux(field, elements):
        if isinstance(elements, List):
            tracker[field].extend(elements)
        else:
            tracker[field].extend([elements] * n)

    aux("feature", feature)
    for k, v in kwargs.items():
        aux(k, v)
    for k in missing_fields:
        aux(k, -1)
    return tracker


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
