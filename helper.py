"""Helper scintific module
Module serves for custom methods to support Customer Journey Analytics Project
"""

# IMPORTS
# -------

# Standard libraries
import ipdb

# 3rd party libraries
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns


# MODULE FUNCTIONS
# ----------------

def get_dissimilarity(df, normalize=True):
    '''Calculates dissimilarity of observations from average
       observation.

    Args:
        df: Data as dataframe of shape (# observations, # variables)

    Returns:
        ser: Calculated dissimilrity as series of size (# observations)
    '''

    # normalize data
    if normalize:
        df_scaled = StandardScaler().fit_transform(df)
        df = pd.DataFrame(df_scaled, columns=df.columns, index=df.index)
    else:
        raise Exception('Not implemented')

    # calculate multivariate dissimilarity
    diss = ((df - df.mean())**2).sum(axis=1)**(1/2)
    return diss


def split_data(df, diss_var, dataset_names, threshold, dis_kws={}, **split_kws):
    '''Function randomly splits data into two sets, calates multivariate
       dissimilarity and keep all oultiers determined by dissimilarity
       treshold in each set.

       Args:
           df: Data as dataframe of shape (# samles, # features)
           diss_var: Names of variables to calculate dissimilarity measure
                     as list of strings
           dataset_names: Names of datasets as list of strings
           threshold: Threshold for dissimilarity measure
                      to determine outliers as float
           dis_kws: Key word arguments of dissimilarity function as dictionary
           split_kws: Key word arguents of train_test_split function

        Returns:
            datasets: Dictionary of splitted datasets as dataframe
    '''

    # calculate dissimilarity series
    dis_kws['normalize'] = (True if 'normalize' not in dis_kws
                            else dis_kws['normalize'])

    dissimilarity = get_dissimilarity(df[diss_var], dis_kws['normalize'])

    # Pop outlier customers
    ext_mask = (dissimilarity > threshold)
    X_ext = df.loc[ext_mask]
    X = df.loc[~ext_mask]

    # drop one random sample to keep even samples in dataset
    # for purpose of having same number of samples after splitting
    if X.shape[0] % 2 != 0:
        split_kws['random_state'] = (1 if 'random_state' not in split_kws
                                     else split_kws['random_state'])
        remove_n = 1
        drop_indices = (X.sample(remove_n,
                                 random_state=split_kws['random_state'])
                         .index)
        X = X.drop(drop_indices)

    # Random split of sample in two groups
    Xa, Xb = train_test_split(X, **split_kws)
    datasets = [Xa, Xb]

    # add outliers to each group
    datasets = {dataset_name: dataset
        for dataset_name, dataset in zip(dataset_names, datasets)}

    for name, dataset in datasets.items():
        datasets[name] = dataset.append(X_ext)

    return datasets


def analyze_cluster_solution(df, vars_, labels, **kws):
    '''Analyzes cluster solution. Following analyses are done:
       1) Hypothesis testing of clusters averages difference
           a) One way ANOVA
           b) ANOVA assumptions
               - residuals normality test: Shapiro-Wilk test
               - equal variances test: Leven's test
           c) Kruskal-Wallis non parametric test
           d) All-Pair non parametric test, Conover test by default
       2) Cluster profile vizualization
       3) Cluster scatterplot vizualization

    Args:
        df: Dataset as pandas dataframe
            of shape(# observations, # variables)
        vars_: Clustering variables as list of strings
        labels: Variable holding cluster labels as string
        kws: Key words arguments of post-hoc test

    Returns:
        summary: Dataframe of hypothesis tests
        post_hoc: List of post_hoc test for each clustering variable
        prof_ax: Axes of profile vizualization
        clst_pg: PairGrid of cluster vizulization
    '''

    def color_not_significant_red(val, signf=0.05):
        '''Takes a scalar and returns a string withthe css property
        `'color: red'` for non significant p_value
        '''
        color = 'red' if val > signf else 'black'
        return 'color: %s' % color

    # get number of seeds
    num_seeds = len(df.groupby(labels).groups)

    # run tests
    kws['post_hoc_fnc'] = (sp.posthoc_conover if 'post_hoc_fnc' not in kws
                          else kws['post_hoc_fnc'])

    summary, post_hoc = profile_cluster_labels(
    df, labels, vars_, **kws)

    # print hypothesis tests
    str_ = 'PROFILE SUMMARY FOR {}'.format(labels.upper())
    print(str_ + '\n' + '-' * len(str_) + '\n')

    str_ = 'Hypothesis testing of clusters averages difference'
    print(str_ + '\n' + '-' * len(str_))

    display(summary.round(2))

    # print post-hoc tests
    str_ = '\nPost-hoc test: {}'.format(kws['post_hoc_fnc'].__name__)
    print(str_ + '\n' + '-' * len(str_) + '\n')

    for var in post_hoc:
        print('\nclustering variable:', var)
        display(post_hoc[var].round(2)
                .style.applymap(color_not_significant_red))

    # print profiles
    str_ = '\nProfile vizualization'
    print(str_ + '\n' + '-' * len(str_))

    prof_ax = (df
               .groupby(labels)
               [vars_]
               .mean()
               .transpose()
               .plot(title='Cluster Profile')
              )
    plt.ylabel('Standardized scale')
    plt.xlabel('Clustering variables')
    plt.show()

    # print scatterplots
    str_ = '\nClusters vizualization'
    print(str_ + '\n' + '-' * len(str_))
    clst_pg = sns.pairplot(x_vars=['recency', 'monetary'],
                           y_vars=['frequency', 'monetary'],
                           hue=labels, data=df, height=3.5)
    clst_pg.set(yscale='log')
    clst_pg.axes[0, 1].set_xscale('log')
    clst_pg.fig.suptitle('Candidate Solution: {} seeds'
                         .format(num_seeds), y=1.01)
    plt.show()

    return summary, post_hoc, prof_ax, clst_pg


def profile_cluster_labels(df, group, outputs, post_hoc_fnc=sp.posthoc_conover):
    '''Test distinctiveness of cluster (group) labes across clustering (output)
    variables using one way ANOVA, shapiro_wilk normality test,
    leven's test of equal variances, Kruskla-Wallis non parametric tests and
    selected all-pairs post hoc test for each output variables.

    Args:
        df: Data with clustering variables and candidate solutions
            as dataframe of shape (# samples, # of variables +
            candidate solutions)

        group: group variables for hypothesis testing as string
        output: output variables for hypothesis testing as list of string
    Returns:
        results: Dataframe of hypothesis tests for each output
    '''

    # initiate summmary dataframe
    summary = (df.groupby(group)[outputs]
                    .agg(['mean', 'median'])
                    .T.unstack(level=-1)
                    .swaplevel(axis=1)
                    .sort_index(level=0, axis=1))
    # initiate posthoc dictionary
    post_hoc = {}

    # cycle over ouptputs
    for i, output in enumerate(outputs):

        # split group levels
        levels = [df[output][df[group] == level]
                  for level in df[group].unique()]

        # calculate F statistics and p-value
        _, summary.loc[output, 'anova_p'] = stats.f_oneway(*levels)

        # calculate leven's test for equal variances
        _, summary.loc[output, 'levene_p'] = stats.levene(*levels)

        # check if residuals are normally distributed by shapiro wilk test
        model = ols('{} ~ C({})'.format(output, group), data=df).fit()
        _, summary.loc[output, 'shapiro_wilk_p'] = stats.shapiro(model.resid)

        # calculate H statistics and p-value for Kruskal Wallis test
        _, summary.loc[output, 'kruskal_wallis_p'] = stats.kruskal(*levels)

        # multiple comparison Conover's test
        post_hoc[output] = post_hoc_fnc(
            df, val_col=output, group_col=group) #, p_adjust ='holm')

    return summary, post_hoc

def get_missmatch(**kws):
    '''
    Cross tabulates dataframe on 2 selected columns and
    calculates missmatch proportion of rows and total

    Args:
        kws: Key word arguments to pd.crosstab function

    Returns:
        crosst_tab: result of cross tabulation as dataframe
        missmatch_rows: missmatch proportion by rows as series
        total_missmatch: total missmatch proportion as float

    '''

    cross_tab = pd.crosstab(**kws)
    missmatch_rows = (cross_tab.sum(axis=1) - cross_tab.max(axis=1))
    total_missmatch = missmatch_rows.sum() / cross_tab.sum().sum()
    missmatch_rows = missmatch_rows / cross_tab.sum(axis=1)
    missmatch_rows.name = 'missmatch_proportion'

    return cross_tab, missmatch_rows, total_missmatch
