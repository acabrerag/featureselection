import gc
import json
import multiprocessing as mp
import os
from datetime import date
from itertools import chain
from typing import List

import numpy as np
import pandas as pd
from fklearn.training.transformation import onehot_categorizer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tsfresh.utilities.dataframe_functions import impute, impute_dataframe_zero

from utilsfun import corr_matrix, roc_weighted, corr_xy, chi, ks, check_dtype, read_csv_batch
from utilsfun import get_numeric, get_caps_floors
from feature_selector import FeatureSelector


today = date.today().strftime("%Y-%m-%d")


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
            selected_features=features
            removed_features=[]
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
                           step_6=[best_dict[k] for k in old_selection], step_6_threshold=res["thresholds"]["lower_best"])
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
            res_dict={}
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


def klar_feature_selection(params: dict, df: pd.DataFrame = None):
    """
    Function to perform feature selection
    """
    """
    Reading the json with the parameters
    dir: directory with the dataset
    name: dataset name
    output_dir: directory for the output files
    chunksize: number of columns to split the dataset and process it, 
                default is -1 and that means the whole dataset will be used in a single split 
    n_jobs: number of cores to use
    missing_breaks: list of breaks to split the interval (0,1) representing the missing value percentage
                    the script create the missing_ranges to analyze the features based on the missing_breaks
    forbidden_features: List of feature to not be considered in the feature selection process
    target: target column name
    basic_features: List of automatically selected features, they are excluded in the feature selection process
                    and will be put in the final result
    index_featues: Features need it in each split
    final_name: name of the final dataset
    Returns
    """
    tracker = new_tracker(["feature", "type", "step", "removed", "step_2", "step_5",
                           "step_6", "step_2_threshold", "step_5_threshold", "step_6_threshold"])
    input_dir = params.get("dir", "")
    output_dir = params.get("output_dir", input_dir)
    n_jobs = int(params.get("n_jobs", 2))
    missing_breaks = params.get("missing_breaks", [x * 0.1 for x in range(1, 10)])
    max_missing_percentage = params.get("max_missing_percentage", 0.8)
    target = params.get("target", "output")
    forbidden_features = params.get("forbidden_features", [])
    index_features = params.get("index_features", ["id_loan", "vintage"])
    basic_features = params.get("basic_features", [])
    final_name = params.get("final_name", "ds_selected")
    missing_ranges = [(-0.1, missing_breaks[0])] + [(e, missing_breaks[n + 1]) for n, e in
                                                    enumerate(missing_breaks[:-1])] + [(missing_breaks[-1], 1)]

    path_dir = "{0}/splits".format(output_dir)
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

    if df is None:
        df = read_csv_batch("{0}/{1}".format(input_dir, params.get("name", "remove_low_information.csv")))
    labels = df[target].replace(params["good"], 0).replace(params["bad"], 1)

    basic_features = list(set(basic_features) & set(df.columns.to_list()))
    df_basic = df[index_features + basic_features]
    df = df[set(df.columns) - set(forbidden_features)]
    # Remove the basic features from the dataframe
    df = df[df.columns.difference(basic_features)]
    feats = list(set(df.columns) - set([target]))
    number_columns, object_columns = get_numeric(df[feats])
    print("Step 1: remove for single unique value")
    # Step 1: remove for single unique value
    fs = FeatureSelector(data=df, random_state=0)
    fs.identify_single_unique()
    df = fs.remove(methods=["single_unique"], keep_one_hot=False)

    old_selection = feats
    removed_features = fs.removed_features
    selected_features = list(set(old_selection) - set(removed_features))

    update_tracker(tracker=tracker, feature=old_selection,
                   type=["numerical" if x in number_columns else "categorical" for x in old_selection]
                   , step=1, removed=[1 if x in removed_features else 0 for x in old_selection])

    # Step 2: remove for missing percentage threshold
    print("Step 2: remove for missing percentage threshold")
    fs = FeatureSelector(data=df, random_state=0)
    fs.identify_missing(missing_threshold_min=-0.1, missing_threshold_max=max_missing_percentage)
    missing_fraction = fs.missing_stats.to_dict()["missing_fraction"]
    df = fs.remove(methods=['missing'], keep_one_hot=False)
    old_selection = selected_features
    removed_features = fs.removed_features
    selected_features = list(set(old_selection) - set(removed_features))
    update_tracker(tracker=tracker, feature=old_selection,
                   type=["numerical" if x in number_columns else "categorical" for x in old_selection]
                   , step=2, removed=[1 if x in removed_features else 0 for x in old_selection],
                   step_2=[missing_fraction[k] for k in old_selection], step_2_threshold=max_missing_percentage)

    for (t_min, t_max) in list(filter(lambda x: x[1] <= max_missing_percentage, missing_ranges)):
        print("Step 3: Remove zero importance for ({0},{1})".format(t_min, t_max))
        fs = FeatureSelector(data=df, random_state=0)
        fs.identify_missing(missing_threshold_min=t_min, missing_threshold_max=t_max)
        ds = fs.remove(methods=['missing'], keep_one_hot=False)
        if ds.shape[1] > 1:
            # Step 3: Remove zero importance
            #todo review zero importance
            fs = FeatureSelector(data=ds, labels=labels, random_state=0)
            fs.identify_zero_importance("classification",
                                        **params.get("importances", {}).get("identify_zero_importance", {}))

            # Keep the feature if one of its dummies is saved
            oh = fs.one_hot_features
            ds_aux = fs.remove(methods=['zero_importance'], keep_one_hot=False)
            removed_oh = set(["_".join(f.split("_")[:-1]) for f in list(set(fs.removed_features) & set(oh))])
            keep_oh = set(["_".join(f.split("_")[:-1]) for f in set(oh) & set(ds_aux.columns.to_list())])
            del ds_aux
            gc.collect()
            old_selection = list(ds.columns)
            removed_features = list(removed_oh - keep_oh) + list(set(fs.removed_features) - set(oh))
            selected_features = list(set(old_selection) - set(removed_features))

            update_tracker(tracker=tracker, feature=old_selection,
                           type=["numerical" if x in number_columns else "categorical" for x in old_selection]
                           , step=3, removed=[1 if x in removed_features else 0 for x in old_selection])

            ds = ds[selected_features]
            print("Step 4: Remove low importance for ({0},{1})".format(t_min, t_max))
            #Todo review low importance
            fs = FeatureSelector(data=ds, labels=labels, random_state=0)
            fs.identify_zero_importance(task="classification",
                                        **params.get("importances", {}).get("identify_low_importance", {}).get(
                                            "identify_zero_importance", {}))
            fs.identify_low_importance(cumulative_importance=params.get("importances", {}).get(
                "identify_low_importance", {}).get(
                "cumulative_importance", 0.95))

            # Step 4: remove for low importance, cumulative importance
            # Keep the feature if one of its dummies is saved
            oh = fs.one_hot_features
            ds_aux = fs.remove(methods=['zero_importance', 'low_importance'], keep_one_hot=False)
            removed_oh = set(["_".join(f.split("_")[:-1]) for f in list(set(fs.removed_features) & set(oh))])
            keep_oh = set(["_".join(f.split("_")[:-1]) for f in set(oh) & set(ds_aux.columns.to_list())])
            del ds_aux
            gc.collect()
            old_selection = selected_features
            removed_features = list(removed_oh - keep_oh) + list(set(fs.removed_features) - set(oh))
            selected_features = list(set(old_selection) - set(removed_features))

            update_tracker(tracker=tracker, feature=old_selection,
                           type=["numerical" if x in number_columns else "categorical" for x in old_selection]
                           , step=4, removed=[1 if x in removed_features else 0 for x in old_selection])

            ds = ds[selected_features]
            if ds.shape[1] > 0:
                ds.to_csv("{0}/reduced_{1}_{2}.csv".format(path_dir, t_min, t_max), index=False)
            else:
                print("Not selected features in ({0},{1})".format(t_min, t_max))
        else:
            print("There is not features to be considered in ({0},{1})".format(t_min, t_max))

    """
    merge the datasets (missing percentage ranges) in a single one for the feature selection considering the max_missing_percentage
    """
    dfs = [df_basic[index_features]]

    for arch in list(map(lambda y: "reduced_{0}_{1}.csv".format(y[0], y[1]),
                         filter(lambda x: x[1] <= max_missing_percentage, missing_ranges))):
        if os.path.isfile("{0}/{1}".format(path_dir, arch)):
            dfs.append(pd.read_csv("{0}/{1}".format(path_dir, arch), low_memory=False))
    df_full = pd.concat(dfs, axis=1)
    del dfs
    gc.collect()
    df_full = df_full.loc[:, ~df_full.columns.duplicated()]
    df_full = df_full.drop(columns=[col for col in df_full.columns if "id_loan." in col])
    df_full[target] = labels
    df_full.to_csv("{0}/ds_full.csv".format(path_dir), index=False)

    """
    Use the StatisticalSelector class for feature selection.
    """
    df_full = df_full.replace([np.inf, -np.inf], np.nan)
    features = list(set(df_full.columns) - set(forbidden_features))
    numeric_features = [col for col in features if df_full[col].dtype not in ['object']]
    impute_method = params.get("impute_method", "zero")
    df_full[numeric_features] = impute(
        df_full[numeric_features]) if impute_method == "median" else impute_dataframe_zero(
        df_full[numeric_features])
    ss = StatisticalSelector(df_full, target, params.get("random_state", 0))
    res = ss.select_features(features, n_jobs=n_jobs, **params.get("select_features", {}))
    tracker2 = ss.tracker

    """
    Save the result: 
        list with features for good/bad/best prediction and type numerical/categorial
        dataframe with the selected features + basic features
    """
    features = res["selected"]["best"]
    for k in ["good", "bad", "best"]:
        with open('{0}/features_{1}.txt'.format(output_dir, k), 'w') as filehandle:
            for listitem in res["selected"][k]:
                filehandle.write('%s\n' % listitem)

    with open("{0}/result.json".format(output_dir), 'w') as json_file:
        json.dump(res["all"], json_file)
    json_file.close()

    update_tracker(tracker=tracker, **tracker2)

    update_tracker(tracker=tracker, feature=features,
                   type=["numerical" if x in number_columns else "categorical" for x in features]
                   , step=8, removed=0)

    tracker["date"] = today
    tracker["model"] = params.get("model", "risk_model")

    features = list(set(features + basic_features)) + [target] + index_features
    del df_full
    gc.collect()
    df = read_csv_batch("{0}/{1}".format(input_dir, params.get("name", "remove_low_information.csv")),
                        usecols=features,
                        low_memory=False)

    df[features].to_csv("{0}/{1}.csv".format(output_dir, final_name), index=False)

    with open("{0}/tracker.json".format(output_dir), 'w') as json_file:
        json.dump(tracker, json_file)
    json_file.close()
    pd.DataFrame(tracker).to_csv("{0}/tracker.csv".format(output_dir), index=False)
    return {"selected": features, "basic": basic_features, "track": tracker}


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
