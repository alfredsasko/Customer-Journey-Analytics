"""Helper scintific module
Module serves for custom methods to support Customer Journey Analytics Project
"""

# IMPORTS
# -------

# Standard libraries
import re
import ipdb
import string

# 3rd party libraries
from google.cloud import bigquery

import numpy as np
import pandas as pd

import nltk
# nltk.download(['wordnet', 'stopwords'])
STOPWORDS = nltk.corpus.stopwords.words('english')

from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scikit_posthocs as sp

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import ComplementNB
from sklearn.metrics import f1_score

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

def query_product_info(client, query_params):
    '''Query product information from bigquery database.
    Distinct records of product_sku, product_name,
    product_brand, product_brand_grp,
    product_category, product_category_grp,
    Args:
        client: Instatiated bigquery.Client to query distinct product
                description(product_sku, product_name, product_category,
                product_category_grp)
        query_params: Query parameters for client
    Returns:
        product_df: product information as distict records
                    as pandas dataframe (# records, # variables)
    '''

    # Check arguments
    # ----------------
    assert isinstance(client, bigquery.Client)
    assert isinstance(query_params, list)

    # Query distinct products descriptions
    # ------------------------------------
    query='''
    SELECT DISTINCT
        hits_product.productSku AS product_sku,
        hits_product.v2productName AS product_name,
        hits_product.productBrand AS product_brand,
        hits.contentGroup.contentGroup1 AS product_brand_grp,
        hits_product.v2productCategory AS product_category,
        hits.contentGroup.contentGroup2 AS product_category_grp
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

    return df

def reconstruct_brand(product_sku, df):
    '''Reconstructs brand from product name and brand variables
    Args:
        product_sku: product_sku as of transaction records on product level
                     of size # transactions on produc level
        df: Product information as output of
            helper.query_product_info in form of dataframe
            of shape (# of distinct records, # of variables)

    Returns:
        recon_brand: reconstructed brand column as pandas series
                     of size # of transactions
    '''

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
                   ['recon_brand'])

    return recon_brand

def reconstruct_category(product_sku, df, category_spec):
    '''Reconstructs category from category variables and product names.

    Args:
        product_sku: product_sku from transaction records on product level
                     of size # transactions on product level
        df: Product information as output of
            helper.query_product_info in form of dataframe
            of shape (# of distinct records, # of variables)

        category_spec: Dictionary with keys as category variable names
                       and values as mappings between category variable levels
                       to category labels in form of dataframe

    Returns:
        recon_category: reconstructed category column as pandas series
                        of size # of trasactions on product level
        category_df: mappings of unique sku to category labels
    '''

    # Check arguments
    # ----------------
    assert isinstance(product_sku,  pd.Series)
    assert isinstance(df, pd.DataFrame)
    assert isinstance(category_spec, dict)

    # reconstruct category name from product name for each sku
    # --------------------------------------------------------

    def get_category_representation(category_label, valid_categories):
        '''Handle multiple categories assigned to one sku.
        For ambigious categories returns missing value.

        Args:
            category_label: Series of category labels for
                            particular sku
            valid_categories: Index of valid unique categories
        Returns:
            label: valid category label or missing value
        '''
        label = valid_categories[valid_categories.isin(category_label)]
        if label.empty or label.size > 1:
            return np.nan
        else:
            return label[0]


    def label_category_variable(df, category_var, label_spec):
        '''reconstruct category labels from category variable.

        Args:
            df: Product information dataframe.
            category_var: Name of category variabel to reconstruct labels
            label_spec: Label mapping between category variable levels
                        and labels.

        Returns:
            var_label: Label mapping to sku as dataframe

        '''

        valid_categories = pd.Index(label_spec
                                    .groupby(['category_label'])
                                    .groups
                                    .keys())

        var_label = (pd.merge(df[['product_name', category_var]]
                                .drop_duplicates(),
                                label_spec,
                                how='left',
                                on=category_var)
                     [['product_name', 'category_label']]
                     .groupby('product_name')
                     ['category_label']
                     .apply(get_category_representation,
                            valid_categories=valid_categories)
                     .reset_index())

        return var_label

    def screen_fit_model(data):
        '''Screens Naive Bayes Classifiers and selects best model
        based on f1 weigted score. Returns fitted model and score.

        Args:
            data: Text and respective class labels as dataframe
                  of shape (# samples, [text, labels])

        Returns:
            model: Best fitted sklearn model
            f1_weighted_score: Test f1 weighted score

        Note: Following hyperparameters are tested
              Algorithm: MultinomialNB, ComplementNB
              ngrams range: (1, 1), (1, 2), (1, 3)
              binarization: False, True
        '''

        # vectorize text inforomation in product_name
        def preprocessor(text):
            # not relevant words
            not_relevant_words = ['google',
                                  'youtube',
                                  'waze',
                                  'android']

            # transform text to lower case and remove punctuation
            text = ''.join([word.lower() for word in text
                            if word not in string.punctuation])

            # tokenize words
            tokens = re.split('\W+', text)

            # Drop not relevant words and lemmatize words
            wn = nltk.WordNetLemmatizer()
            text = ' '.join([wn.lemmatize(word) for word in tokens
                    if word not in not_relevant_words + STOPWORDS])

            return text

        # define pipeline
        pipe = Pipeline([('vectorizer', CountVectorizer()),
                         ('classifier', None)])

        # define hyperparameters
        param_grid = dict(vectorizer__ngram_range=[(1, 1), (1, 2), (1, 3)],
                          vectorizer__binary=[False, True],
                          classifier=[MultinomialNB(),
                                      ComplementNB()])

        # screen naive buyes models
        grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=5,
                                   scoring='f1_weighted', n_jobs=-1)

        # devide dataset to train and test set using stratification
        # due to high imbalance of lables frequencies
        x_train, x_test, y_train, y_test = train_test_split(
            data['product_name'],
            data['recon_category'],
            test_size=0.25,
            stratify=data['recon_category'],
            random_state=1)

        # execute screening and select best model
        grid_search.fit(x_train, y_train)

        # calculate f1 weighted test score
        y_pred = grid_search.predict(x_test)
        f1_weigted_score = f1_score(y_test, y_pred, average='weighted')

        return grid_search.best_estimator_, f1_weigted_score

    # reconstruct category label from cateogry variables
    recon_labels = dict()
    for var, label_spec in category_spec.items():
        recon_labels[var] = (label_category_variable(df, var, label_spec)
                             .set_index('product_name'))

    recon_labels['product_category'][
        recon_labels['product_category'].isna()
    ] = recon_labels['product_category_grp'][
        recon_labels['product_category'].isna()
    ]

    # reconstruct category label from produc names
    valid_categories = pd.Index(category_spec['product_category_grp']
                        .groupby(['category_label'])
                        .groups
                        .keys())

    category_df = (pd.merge(df[['product_sku', 'product_name']]
                            .drop_duplicates(),
                            recon_labels['product_category'],
                            how='left',
                            on = 'product_name')
                   [['product_sku', 'product_name', 'category_label']]
                   .groupby('product_sku')
                   .agg({'product_name': lambda name: name.str.cat(sep=' '),
                         'category_label': lambda label:
                         get_category_representation(label, valid_categories)})
                   .reset_index())

    category_df.rename(columns={'category_label': 'recon_category'},
                       inplace=True)

    # associate category from category names and variables on transaction data
    # ------------------------------------------------------------------------
    recon_category = (pd.merge(product_sku.to_frame(),
                           category_df[['product_sku', 'recon_category']],
                           on='product_sku',
                           how='left')
                  )

    # predict category of transactions where category is unknown
    # Multinomial and Complement Naive Bayes model is screened
    # and finetuned using 1-grams, 2-grams and 3-grams
    # as well as binarization (Tru or False)
    # best model is selected based on maximizing test f1 weigted score
    # ----------------------------------------------------------------

    # screen best model and fit it on training data
    model, f1_weighted_score = screen_fit_model(
        category_df[['product_name', 'recon_category']]
        .dropna()
        )

    # predict category labels if model has f1_weighted_score > threshold
    f1_weighted_score_threshold = 0.8
    if f1_weighted_score < f1_weighted_score_threshold:
        raise Exception(
        'Accuracy of category prediction below threshold {:.2f}'
        .format(f1_weighted_score_threshold))
    else:
        product_name = (pd.merge(recon_category
                               .loc[recon_category['recon_category'].isna(),
                                    ['product_sku']],
                               category_df[['product_sku', 'product_name']],
                               how='left',
                               on='product_sku')
                      ['product_name'])

        category_label = model.predict(product_name)
        recon_category.loc[recon_category['recon_category'].isna(),
                       'recon_category'] = category_label

    return recon_category['recon_category']

def reconstruct_sales_region(subcontinent):
    '''Reconstruct sales region from subcontinent'''

    if (pd.isna(subcontinent)
        or subcontinent.lower() == '(not set)'):
        sales_region = np.nan

    elif ('africa' in subcontinent.lower()
          or 'europe' in subcontinent.lower()):
        sales_region = 'EMEA'

    elif ('caribbean' in subcontinent.lower()
          or subcontinent.lower() == 'central america'):
        sales_region = 'Central America'

    elif subcontinent.lower() == 'northern america':
        sales_region = 'North America'

    elif subcontinent.lower() == 'south america':
        sales_region = 'South America'

    elif ('asia' in subcontinent.lower()
          or subcontinent.lower() == 'australasia'):
        sales_region = 'APAC'

    else:
        raise Exception(
            'Can not assign sales region to {} subcontinent'
            .format(subcontinent))

    return sales_region

def reconstruct_traffic_keyword(text):
    '''Reconstructs traffic keywords to more simple representation'''

    if pd.isna(text):
        text = '(not applicable)'

    elif ((text != '(not provided)')
          and (re.search('(\s+)', text) is not None)):

            # transform text to lower case and remove punctuation
            text = ''.join([word.lower() for word in text
                            if word not in string.punctuation.replace('/', '')])

            # tokenize words
            tokens = re.split('\W+|/', text)

            # Drop not relevant words and lemmatize words
            wn = nltk.WordNetLemmatizer()
            text = ' '.join([wn.lemmatize(word) for word in tokens
                    if word not in STOPWORDS])

    return text


def aggregate_data(df):
    '''Encode and aggregate engineered and missing value free data
    on client level

    Args:
        df: engineered and missing value free data as
            pandas dataframe of shape (# transaction items, # variables)

        agg_df: encoded and aggregated dataframe
                of shape(# clients, # encoded & engineered variables)
                with client_id index
    '''
    # identifiers
    id_vars = pd.Index(
        ['client_id',
         'session_id',
         'transaction_id',
         'product_sku']
    )

    # session variables
    session_vars = pd.Index(
        ['visit_number',           # avg_visits
         'date',                   # month, week, week_day + one hot encode + sum
         'pageviews',              # avg_pageviews
         'time_on_site',           # avg_time_on_site
         'ad_campaign',            # sum
         'source',                 # one hot encode + sum
         'traffic_keyword',        # one hot encode + sum
         'browser',                # one hot encode + sum
         'operating_system',       # one hot encode + sum
         'device_category',        # one hot encode + sum
         'continent',              # one hot encode + sum
         'subcontinent',           # one hot encode + sum
         'country',                # one hot encode + sum
         'sales_region',           # one hot encode + sum
         'social_referral',        # sum
         'social_network',         # one hot encode + sum
         'channel_group']          # one hot encode + sum
    )

    # group session variables from item to session level
    session_df = (df[['client_id',
                      'session_id',
                      *session_vars.to_list()]]
                  .drop_duplicates()

                   # drop ambigious region 1 case
                  .drop_duplicates(subset='session_id'))

    # reconstruct month, weeek and week day variables
    session_df['month'] = session_df['date'].dt.month
    session_df['week'] = session_df['date'].dt.week
    session_df['week_day'] = session_df['date'].dt.weekday + 1
    session_df = session_df.drop(columns='date')

    # encode variables on session level
    keep_vars = [
        'client_id',
        'session_id',
        'visit_number',
        'pageviews',
        'time_on_site',
        'social_referral',
        'ad_campaign'
    ]

    encode_vars = session_df.columns.drop(keep_vars)
    enc_session_df = pd.get_dummies(session_df,
                                    columns=encode_vars.to_list(),
                                    prefix_sep='*')

    # remove not relevant encoded variables
    enc_session_df = enc_session_df.drop(
        columns=enc_session_df.columns[
            enc_session_df.columns.str.contains('not set|other')
        ]
    )

    # summarize session level variables on customer level
    sum_vars = (pd.Index(['social_referral', 'ad_campaign'])
                .append(enc_session_df
                        .columns
                        .drop(keep_vars)))

    client_session_sum_df = (enc_session_df
                             .groupby('client_id')
                             [sum_vars]
                             .sum())

    client_session_avg_df = (
        enc_session_df
        .groupby('client_id')
        .agg(avg_visits=('visit_number', 'mean'),
             avg_pageviews=('pageviews', 'mean'),
             avg_time_on_site=('time_on_site', 'mean'))
    )

    client_session_df = pd.concat([client_session_avg_df,
                                   client_session_sum_df],
                                  axis=1)


    # product level variables
    product_vars = pd.Index([
        'product_category',        # one hot encode + sum
        'product_price',           # avg_product_revenue
        'product_quantity',        # avg_product_revenue
        'hour']                    # one hot encoded + sum
    )

    avg_vars = pd.Index([
        'product_price',
        'product_quantity'
    ])

    sum_vars = pd.Index([
        'product_category',
        'hour'
    ])

    enc_product_df = pd.get_dummies(df[id_vars.union(product_vars)],
                                    columns=sum_vars,
                                    prefix_sep='*')

    # summarize product level variables on customer level
    client_product_sum_df = (enc_product_df
                             .groupby('client_id')
                             [enc_product_df.columns.drop(avg_vars)]
                             .sum())

    def average_product_vars(client):
        d = {}
        d['avg_product_revenue'] = ((client['product_price']
                                     * client['product_quantity'])
                                    .sum()
                                    / client['product_quantity'].sum())

        d['unique_products'] = len(client['product_sku'].unique())

        return pd.Series(d, index=['avg_product_revenue',
                                   'unique_products'])

    client_product_avg_df = (enc_product_df
                             .groupby('client_id')
                             .apply(average_product_vars))

    client_product_df = pd.concat([client_product_avg_df,
                                   client_product_sum_df]
                                  , axis=1)

    agg_df = pd.concat([client_session_df,
                        client_product_df],
                       axis=1)

    return agg_df
