"""Helper scintific module
Module serves for custom methods to support Customer Journey Analytics Project
"""

# IMPORTS
# -------

# Standard libraries
import re
import ipdb

# 3rd party libraries
from google.cloud import bigquery

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

def reconstruct_brand(product_sku, client, query_params):
    '''Reconstructs brand from product name and brand variables
    Args:
        product_sku: product_sku as series of size (# transactions, 2)
        client: Instatiated bigquery.Client to query distinct product
                description(product_sku, product_name, product_brand,
                product_brand_grp)
        query_params: Query parameters for client

    Returns:
        recon_brand: reconstructed brand column as pandas dataframe
                     of shape (# transactions, ['product_sku', 'recon_brand'])
    '''

    # Check arguments
    # ----------------
    assert isinstance(product_sku,  pd.Series)
    assert isinstance(client, bigquery.Client)

    # Query distinct products descriptions
    # ------------------------------------
    query='''
    SELECT DISTINCT
        hits_product.productSku AS product_sku,
        hits_product.v2productName AS product_name,
        hits_product.productBrand AS product_brand,
        hits.contentGroup.contentGroup1 AS product_brand_grp
    FROM
        `bigquery-public-data.google_analytics_sample.ga_sessions_*`
        LEFT JOIN UNNEST(hits) AS hits
        LEFT JOIN UNNEST(hits.product) AS hits_product
    WHERE
        _TABLE_SUFFIX BETWEEN @start_date AND @end_date
        AND hits_product.productSku IS NOT NULL
    ORDER BY
        product_sku
    '''

    job_config = bigquery.QueryJobConfig()
    job_config.query_parameters = query_params
    df = client.query(query, job_config=job_config).to_dataframe()


    # predict brand name from product name for each sku
    # -------------------------------------------------

    # valid brands
    brands = ['Android',
              'Chrome',
              r'\bGo\b',
              'Google',
              'Google Now',
              'YouTube',
              'Waze']

    # concatenate different product names for each sku
    brand_df = (df[['product_sku', 'product_name']]
                .drop_duplicates()
                .groupby('product_sku')
                ['product_name']
                .apply(lambda product_name: ' '.join(product_name))
                .reset_index()
                )

    # drop (no set) sku's
    brand_df = brand_df.drop(
        index=brand_df.index[brand_df['product_sku'] == '(not set)'])


    # predict brand name from product name for each sku
    brand_df['recon_brand'] = (
        brand_df['product_name']
        .str.extract(r'({})'.format('|'.join(set(brands)),
                     flags=re.IGNORECASE))
    )

    # adjust brand taking account spelling errors in product names
    brand_df.loc[
        brand_df['product_name'].str.contains('You Tube', case=False),
        'recon_brand'
    ] = 'YouTube'


    # predict brand name from brand variables for sku's where
    # brand couldn't be predected from product name
    # --------------------------------------------------------

    # get distinct product_sku and brand variables associations
    brand_vars = ['product_brand', 'product_brand_grp']
    brand_var = dict()
    for brand in brand_vars:
        brand_var[brand] = (df[['product_sku', brand]]
                            .drop(index=df.index[(df['product_sku'] == '(not set)')
                                                 | df['product_sku'].isna()
                                                 | (df[brand] == '(not set)')
                                                 | df[brand].isna()])
                            .drop_duplicates()
                            .drop_duplicates(subset='product_sku', keep=False))

    # check for brand abiguity at sku level
    old_brand = brand_var['product_brand'].set_index('product_sku')
    new_brand = brand_var['product_brand_grp'].set_index('product_sku')
    shared_sku = old_brand.index.intersection(new_brand.index)

    if not shared_sku.empty:

        # delete sku's with abigious brands
        ambigious_sku = shared_sku[
            old_brand[shared_sku].squeeze().values
            != new_brand[shared_sku].squeeze().values
        ]

        old_brand = old_brand.drop(index=ambigious_sku, errors='ignore')
        new_brand = new_brand.drop(index=ambigious_sku, errors='ignore')

        # delete sku's with multiple brands in new_brand
        multiple_sku = shared_sku[
            old_brand[shared_sku].squeeze().values
            == new_brand[shared_sku].squeeze().values
        ]

        new_brand = new_brand.drop(index=multiple_sku, errors='ignore')

    # concatenate all associations of brand variables and product sku's
    brand_var = pd.concat([old_brand.rename(columns={'product_brand':
                                                     'recon_brand_var'}),
                           new_brand.rename(columns={'product_brand_grp':
                                                     'recon_brand_var'})])

    # predict brand name from brand variables
    brand_df.loc[brand_df['recon_brand'].isna(), 'recon_brand'] = (
        pd.merge(brand_df['product_sku'], brand_var, on='product_sku', how='left')
        ['recon_brand_var']
    )

    # recode remaining missing (not set) brands by Google brand
    # ---------------------------------------------------------
    brand_df['recon_brand'] = brand_df['recon_brand'].fillna('Google')

    # predict brand from brand names and variables on transaction data
    # ----------------------------------------------------------------
    recon_brand = (pd.merge(product_sku.to_frame(),
                           brand_df[['product_sku', 'recon_brand']],
                           on='product_sku',
                           how='left')
                   .reindex(product_sku.index)
                   ['recon_brand']
                  )

    return recon_brand
