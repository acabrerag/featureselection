import itertools
import operator
import sys
import warnings
from functools import reduce
from typing import List

import numpy as np
import pandas as pd
import scorecardpy as sc
import statsmodels.discrete.discrete_model as sm
from monotonic_binning import monotonic_woe_binning as bin
from optbinning import OptimalBinning
from optbinning.binning.binning_statistics import BinningTable
from tsfresh.utilities.dataframe_functions import impute, impute_dataframe_zero
from varclushi import VarClusHi
from xverse.transformer import MonotonicBinning

warnings.filterwarnings("ignore")


def monotone_increasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.le, pairs))


def monotone_decreasing(lst):
    pairs = zip(lst, lst[1:])
    return all(itertools.starmap(operator.ge, pairs))


def monotone(lst):
    return monotone_increasing(lst) or monotone_decreasing(lst)


class Selector():

    def __init__(self):
        self.to_keep = []
        self.to_drop = []

    def forward_selection(self, data: pd.DataFrame, target: str, features: List[str], initial_features: List[str] = [],
                          significance_level: float = 0.05):
        best_features = []
        for new_column in features:
            # fit the model
            model = sm.Logit(data[target], data[initial_features + best_features + [new_column]]).fit(full_output=0,
                                                                                                      disp=0)
        # compute the metric
        p_value = model.pvalues[new_column]
        # Eval the condition
        if (p_value < significance_level):
            best_features.append(new_column)
        self.to_keep = initial_features + best_features
        self.to_drop = list(set(features) - set(self.to_keep))
        return self.to_keep

    def backward_elimination(self, data, target, features, initial_features=[], significance_level=0.1):
        order = features
        order.reverse()
        accepted = features
        while (len(accepted) > 0):
            # fit the model
            model = sm.Logit(data[target], data[initial_features + features]).fit(method="nm", full_output=0, disp=0)
            # compute the metric
            p_values = model.pvalues.to_dict()
            accepted = {k: v for k, v in p_values.items() if v >= significance_level}
            accepted = {k: accepted[k] for k in order if k in accepted.keys()}
            # Eval the condition
            if len(accepted) > 0:
                excluded_feature = list(accepted.keys())[0]
                features.remove(excluded_feature)

        self.to_keep = initial_features + features
        self.to_drop = list(set(features) - set(self.to_keep))
        return self.to_keep

    def stepwise_selection(self, data, target, features, initial_features=[], SL_in=0.05, SL_out=0.1):
        # https://towardsdatascience.com/feature-selection-using-wrapper-methods-in-python-f0d352b346f
        best_features = []
        order = features
        order.reverse()
        for new_column in features:
            model = sm.Logit(data[target], data[initial_features + best_features + [new_column]]).fit(method="nm",
                                                                                                      full_output=0,
                                                                                                      disp=0)
            p_value = model.pvalues[new_column]
            if (p_value < SL_in):
                best_features.append(new_column)
                if (len(best_features) > 0):
                    p_values = sm.Logit(data[target], data[initial_features + best_features]).fit(method="nm",
                                                                                                  full_output=0,
                                                                                                  disp=0).pvalues.to_dict()
                    accepted = {k: v for k, v in p_values.items() if v >= SL_out}
                    accepted = {k: accepted[k] for k in order if k in accepted.keys()}
                    if len(accepted) > 0:
                        excluded_feature = list(accepted.keys())[0]
                        best_features.remove(excluded_feature)

        self.to_keep = initial_features + best_features
        self.to_drop = list(set(features) - set(self.to_keep))
        return self.to_keep

    def selection(self, method: str = "stepwise", **args):
        if method == "stepwise":
            return self.stepwise_selection(**args)
        elif method == "forward":
            return self.forward_selection(**args)
        elif method == "backward":
            return self.backward_elimination(**args)
        else:
            raise NotImplementedError('%s type has not been run.' % type)


class ScorecardSelector():
    """
    Class for performing feature selection using information values.
    based on the book: Intelligent credit scoring _ building and implementing better creditrisk scorecards, Naeem Siddiqi
    Implements two functions for feature selection:
        1. filter by information value, missing value percentage and identical value percentage.
        2. selection using a Logistic Regression over woe transformed features.
    Also functions for WOE, group features are implemented.
    """

    def __init__(self, data: pd.DataFrame, target: str, features: list):

        self.data = data
        self.target = target
        self.features = features
        self.to_drop = {}
        self.to_keep = {}

    def filter(self, elements: List[str], iv_limit: float = 0.2, missing_limit: float = 0.5,
               identical_limit: float = 0.95, **kwargs):
        """
        Compute information value (using the Binning class),
        the missing value percentage and identical value percentage
        for all features and filter over the params.
        +-------------------+-----------------------------+
        | Information Value | Variable Predictiveness     |
        +-------------------+-----------------------------+
        | Less than 0.02    | Not useful for prediction   |
        +-------------------+-----------------------------+
        | 0.02 to 0.1       | Weak predictive Power       |
        +-------------------+-----------------------------+
        | 0.1 to 0.3        | Medium predictive Power     |
        +-------------------+-----------------------------+
        | 0.3 to 0.5        | Strong predictive Power     |
        +-------------------+-----------------------------+
        | >0.5              | Suspicious Predictive Power |
        +-------------------+-----------------------------+
        """
        binner = Binning(self.data[elements + [self.target]], self.target, elements)
        binner.woe(**kwargs)
        iv = binner.iv()
        iv_list = pd.DataFrame.from_dict(iv, orient='index', columns=["info_value"]).reset_index().rename(
            columns={'index': 'variable'})
        # -na percentage
        nan_rate = lambda a: a[a.isnull()].size / a.size
        na_perc = self.data[elements].apply(nan_rate).reset_index(name='missing_rate').rename(
            columns={'index': 'variable'})
        # -identical percentage
        idt_rate = lambda a: a.value_counts().max() / a.size
        identical_perc = self.data[elements].apply(idt_rate).reset_index(name='identical_rate').rename(
            columns={'index': 'variable'})
        # dataframe iv na idt
        dt_var_selector = iv_list.merge(na_perc, on='variable').merge(identical_perc, on='variable')

        self.to_drop.update({"iv": list(dt_var_selector.query('(info_value <= {})'.format(iv_limit))["variable"])})
        self.to_drop.update(
            {"mr": list(dt_var_selector.query('(missing_rate >= {})'.format(missing_limit))["variable"])})
        self.to_drop.update(
            {"ir": list(dt_var_selector.query('(identical_rate >= {})'.format(identical_limit))["variable"])})
        self.to_drop.update({"cant_split": list(set(elements) - set(dt_var_selector["variable"]))})
        self.to_keep = list(set(elements) - set(list(reduce(lambda x, y: x + y, self.to_drop.values()))))
        return self.to_keep

    def grouping(self, imputation: str = "impute"):
        """
            Create groups for numerical features based on a hierarchical structure with PCA using VarClusHi class
        """
        if imputation == "impute":
            df = impute(self.data[self.features])
        vc = VarClusHi(df)
        vc.varclus()
        clusters = vc.rsquare
        # groups = {k: clusters[clusters["Cluster"] == k]["Variable"].to_list() for k in clusters["Cluster"].unique()}
        groups = {k: {"features": clusters[clusters["Cluster"] == k]["Variable"].to_list(), "type": "numerical"} for k
                  in clusters["Cluster"].unique()}

        return groups

    def convert(self, data: pd.DataFrame, bins: dict, no_cores: int = None):
        """
        Apply woe transformation to the specified data using scorecardpy
        """
        return sc.woebin_ply(data, bins, no_cores=no_cores)

    def iv_group(self, groups, max_elements, **kwargs):
        """
        For each group compute the information value (bins,woe,iv) and apply a filter.
        Results are ordered for descending IV.
        For each group a max number of elements can be used.
        Irrelevant groups are ignored.
        """
        res = {}
        for group, elements in groups.items():
            type = elements["type"]
            elements = elements["features"]
            binner = Binning(self.data[elements + [self.target]], self.target, elements)

            woe = binner.woe(**kwargs, dtype=type)
            iv = binner.iv()
            bin = binner.bins
            methods = kwargs.get("filter_methods",
                                 {"methods": ["IV"] if type == "categorical" else ["IV", "Trend"], "limit": 0.05})
            if methods:
                filter_list = binner.filter(**methods[type])
                self.to_drop.update({"group_limit": filter_list["to_keep"][max_elements:]})
                woe = {k: v for k, v in woe.items() if k in filter_list["to_keep"][0:max_elements]}
                iv = {k: v for k, v in iv.items() if k in filter_list["to_keep"][0:max_elements]}
            if len(iv):
                res.update({group: {"mean": np.mean(list(iv.values())), "order": iv, "woe": woe, "bin": bin}})
            res = dict(sorted(res.items(), key=lambda x: x[1]["mean"], reverse=False))
        return res

    def selection(self, groups=None, type="multiple", selection="stepwise", **kwargs):
        """
        selection using a Logistic Regression over woe transformed features.
        If groups are not specified, we compute the groups (only for numerical features)
        type: "multiple" or  "single".
            "multiple" is for perform a selection in each group
            "single" is for consider a single group
            For more info: see the book.
        selection: "forward","backward","stepwise"
        """
        if groups is None:
            groups = self.grouping()
        groups_iv = self.iv_group(groups=groups, **kwargs)
        groups = {k: list(v["order"].keys()) for k, v in groups_iv.items()}
        if len(groups) > 1:
            features = reduce(lambda x, y: x + y, list(map(lambda x: list(x[1]["order"].keys()), groups_iv.items())))
        else:
            features = list(list(groups_iv.items())[0][1]["order"].keys())
        bins = {}
        for k, v in groups_iv.items():
            bins.update(v["woe"])
        ds = self.convert(self.data[features + [self.target]], bins)
        ds.columns = [col[:-4] if col[-4:] == "_woe" else col for col in ds.columns]
        print("Selection will be performed using {0} features".format(len(features)))
        sel = Selector()
        if type == "multiple":
            initial_features = []
            for group, elements in groups.items():
                r = sel.selection(data=ds, target=self.target, features=elements, method=selection)
                initial_features = initial_features + r

        elif type == "single":
            order = reduce(lambda x, y: x + y, list(map(lambda x: list(x[1]["order"].keys()), groups_iv.items())))
            initial_features = sel.selection(data=ds, target=self.target, features=order, method=selection)
        else:
            raise NotImplementedError('%s type has not been run.' % type)
        return {"selected": initial_features, "not selected": list(set(features) - set(initial_features)),
                "summary": {k: v for k, v in bins.items() if k in initial_features}}


class Binning():
    """
    Class for perform Binning, WOE, IV
    """

    def __init__(self, data: pd.DataFrame, target: str, features: list, random_state=None):

        self.data = data
        self.target = target
        self.features = features
        self.random_state = random_state
        self.bins = None
        self.woes = None
        self.ivs = None

    def binning(self, variables: List[str] = None, method: str = "scorecardpy", dtype: str = "numerical",
                no_cores: int = 7):

        """
        Use auxiliar libraries to compute the breaks for the variables
        method: "optbinning","scorecardpy", "xverse" (Only for numerical features)
        Returns a dict with the breaks for each feature.
        """
        print(f"selected method: {method}")
        output_bins = {}
        if variables is None:
            variables = self.features
        if method == "optbinning":
            for variable in variables:
                optb = OptimalBinning(name=variable, dtype=dtype, solver="cp")
                optb.fit(impute_dataframe_zero(self.data[[variable]])[variable], self.data[self.target])
                if dtype == "numerical":
                    output_bins.update({variable: list(optb.splits)})
                else:
                    output_bins.update({variable: pd.Series(["%,%".join(x.tolist()) for x in optb.splits])})

        elif method == "xverse":
            if dtype == "numerical":
                clf = MonotonicBinning()
                clf.fit([self.data[variables], self.data[self.target].values], None)  # xverse problem X,y = X
                output_bins = clf.bins
            else:
                raise NotImplementedError('%s method has not been run for categorical features.' % method)

        elif method == "scorecardpy":
            print(len(variables), method, no_cores)
            output_bins = sc.woebin(self.data[variables + [self.target]], y=[self.target], no_cores=no_cores)
            output_bins = {k: v["breaks"] for k, v in output_bins.items()}
        elif method == "monotonic_binning":
            if dtype == "numerical":
                for variable in variables:
                    bin_object = bin.Binning(self.target, n_threshold=50, y_threshold=10, p_threshold=0.35, sign=False)
                    bin_object.fit(self.data[[self.target, variable]])
                    output_bins.update({variable: bin_object.bins[1:]})
            else:
                raise NotImplementedError('%s method has not been run for numerical features.' % method)
        else:
            raise NotImplementedError('%s method has not been run.' % method)

        self.bins = output_bins
        return output_bins

    def woe(self, breaks_adj: dict = None, **kwargs):
        """
        For each feature, compute the woe for each bin with the specified breaks.
        Use scorecardpy library.
        Returns a dict with a summary table for each feature-
        """
        no_cores = kwargs.get("no_cores", None)
        if breaks_adj is None:
            if self.bins is None:
                self.binning(no_cores=no_cores)
            breaks_adj = self.bins
        variables = [var for var, val in breaks_adj.items() if len(val) > 0] + [self.target]
        print("{} Features cant be binned".format(len([var for var, val in breaks_adj.items() if len(val) == 0])))
        self.woes = sc.woebin(self.data[variables], y=self.target, breaks_list=breaks_adj, no_cores=no_cores,
                              method=kwargs.get("method",
                                                None))
        return self.woes

    def iv(self):
        """
        Computes the information value for each feature. The calculation use the result of woe().
        Returns a dict with the total information value for each feature
        """
        if self.woes is None:
            self.woes = self.woe()
        self.ivs = {x[0]: x[1]["total_iv"].iloc[0] for x in self.woes.items()}
        self.ivs = dict(sorted(self.ivs.items(), key=lambda x: x[1], reverse=False))
        return self.ivs

    def filter(self, methods: List = ["IV"], limit: str = 0.1):
        """
        Apply filter base on a minimum iv limit and a monotone behaviour.
        """
        to_drop = []
        for method in methods:
            if method == "IV":
                ivs = self.ivs
                to_drop = to_drop + list(filter(lambda x: ivs[x] < limit, ivs.keys()))
            elif method == "Trend":
                to_drop = to_drop + [k for k, v in self.woes.items() if
                                     not monotone(v[v["is_special_values"] == False]["woe"])]
            else:
                raise NotImplementedError('%s method has not been run.' % method)
            to_keep = list(set(self.features) - set(to_drop))
        return {"to_keep": to_keep, "to_drop": list(set(to_drop))}


def report(table, directory, plot=True, analysis=True, **kwargs):
    special = table[table["is_special_values"] == 1]
    not_special = table[table["is_special_values"] != 1]
    dtype = kwargs.get("dtype", "numerical")
    name = table["variable"].iloc[0]
    if dtype == "numerical":
        _splits_optimal = [float(x) for x in list(not_special["breaks"])[0:-1]]
    else:
        _splits_optimal = list(not_special["breaks"])
    if len(special) == 0:
        _special_n_nonevent = [0, 0]
        _special_n_event = [0, 0]
    elif len(special) == 1:
        if special["breaks"].iloc[0] == "missing":
            _special_n_nonevent = [0, special["good"].iloc[0]]
            _special_n_event = [0, special["bad"].iloc[0]]
        else:
            _special_n_nonevent = list(special["good"])
            _special_n_event = list(special["bad"])
    _n_nonevent = np.array(list(not_special["good"]) + _special_n_nonevent)
    _n_event = np.array(list(not_special["bad"]) + _special_n_event)
    _categories = []
    _cat_others = []
    user_splits = []

    tabla = BinningTable(name=name, dtype=dtype, splits=_splits_optimal, n_nonevent=_n_nonevent,
                         n_event=_n_event, categories=_categories, cat_others=_cat_others,
                         user_splits=user_splits)
    tabla.build()

    if plot:
        tabla.plot(savefig="{0}/plots/{1}.jpeg".format(directory, name), **kwargs)
    if analysis:
        sys.stdout = open("{0}/reports/{1}".format(directory, name), "w")
        tabla.analysis()
        sys.stdout.close()
