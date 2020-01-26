import numpy as np
import pandas as pd
from scipy import interp
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns):
        self.ohe_columns_ = []
        self.pre_ohe_columns_ = []
        self.post_ohe_reordered_columns_ = []
        self.categorical_columns = categorical_columns
        self.ohe_prefix = '_ohe_'

    def ohe_(self, df):
        print('Before one-hot-encoding: %s' % str(df.shape))
        assert all(self.ohe_prefix not in col for col in df.columns)
        df_dummies = pd.get_dummies(df, columns=self.categorical_columns, dummy_na=True, prefix_sep=self.ohe_prefix)
        print('After one-hot-encoding: %s' % str(df_dummies.shape))

        return df_dummies

    def fit_transform(self, df, y=None, **fit_params):
        self.pre_ohe_columns_ = df.columns
        df = self.ohe_(df)
        self.ohe_columns_ = [col for col in df.columns if self.ohe_prefix in col]
        df = self.reorder_cols_(df)
        self.post_ohe_reordered_columns_ = df.columns
        return df

    def transform(self, df):
        df = self.ohe_(df)

        for col in df.columns:
            if self.ohe_prefix in col and col not in self.ohe_columns_:
                print('Dropping unexpected column: %s' % col)
                df = df.drop(col, axis=1)

        for col in self.ohe_columns_:
            if col not in df.columns:
                print('One-hot-encoding: adding missing column %s' % col)
                df[col] = np.nan
        df = self.reorder_cols_(df)
        return df

    def fit(self, df):
        df_copy = df.copy()
        _ = self.fit_transform(df_copy)
        return self

    def reorder_cols_(self, df):
        return df[sorted(df.columns)]


def get_df_from_pipe(pipe, array):
    ohe = pipe.named_steps['ohe']
    f_selection = pipe.named_steps['f_selection']
    final_array = pd.DataFrame(array, columns=ohe.post_ohe_reordered_columns_[f_selection.get_support()])
    return final_array


def combine(x, y, aux):
    x = x.reset_index(drop=True)
    y = y.reset_index(drop=True)
    aux = aux.reset_index(drop=True)
    df = pd.concat([x, y], axis=1)
    df = pd.concat([df, aux], axis=1)
    return df

import math


def find_dcg(element_list):
    score = 0.0
    for order, rank in enumerate(element_list):
        score += float(rank)/math.log2((order+2))
    return score


def _order_lists(reference, hypothesis):
    pair_ref_list = sorted([x for x in enumerate(reference)], key=lambda x: x[1])
    mapped_hyp_list = [hypothesis[x[0]] for x in pair_ref_list]

    return [x[1] for x in pair_ref_list], mapped_hyp_list


def find_ndcg(reference, hypothesis):
    return find_dcg(hypothesis)/find_dcg(reference)

def find_rankdcg(reference, hypothesis):

    reference_list, hypothesis_list = _order_lists(reference, hypothesis)

    ordered_list = reference_list[:] # creating ordered list
    ordered_list.sort(reverse=True)

    high_rank = float(len(set(reference_list))) # max rank
    reverse_rank = 1.0            # min score (reversed rank)
    relative_rank_list = [high_rank]
    reverse_rank_list = [reverse_rank]

    for index, rank in enumerate(ordered_list[:-1]):
        if ordered_list[index+1] != rank:
            high_rank -= 1.0
            reverse_rank += 1.0
        relative_rank_list.append(high_rank)
        reverse_rank_list.append(reverse_rank)

    # map real rank to relative rank
    reference_pair_list = [x for x in enumerate(reference_list)]
    sorted_reference_pairs = sorted(reference_pair_list, key=lambda p: p[1], \
                                    reverse=True)
    rel_rank_reference_list = [0] * len(reference_list)
    for position, rel_rank in enumerate(relative_rank_list):
        rel_rank_reference_list[sorted_reference_pairs[position][0]] = rel_rank

    # computing max/min values (needed for normalization)
    max_score = sum([rank/reverse_rank_list[index] for index, rank \
                     in enumerate(relative_rank_list)])
    min_score = sum([rank/reverse_rank_list[index] for index, rank \
                     in enumerate(reversed(relative_rank_list))])

    # computing and mapping hypothesis to reference
    hypothesis_pair_list = [x for x in enumerate(hypothesis_list)]
    sorted_hypothesis_pairs = sorted(hypothesis_pair_list, \
                                     key=lambda p: p[1], reverse=True)
    eval_score = sum([rel_rank_reference_list[pair[0]] / reverse_rank_list[index] \
                      for index, pair in enumerate(sorted_hypothesis_pairs)])

    return (eval_score - min_score) / (max_score - min_score)
    