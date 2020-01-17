# Customer Journey Analytics

Do you want to increase revenue and marketing ROI from your e-commerce platform? If yes, this is the project for you, so read further.

### Table of Contents
1. [Project Motivation](#motivation)
2. [Results](#results)
4. [Installation](#installation)
3. [File Descriptions](#files)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Project Motivation<a name="motivation"></a>

Customer journey mapping has become very popular in recent years. Making it right can upgrade marketing strategy, boost personalized branding and offerings and result in increased revenue and marketing ROI. To bring value to the business, it requires a healthy balance of qualitative knowledge of customer-facing functions about market and customers and quantitative insights, which can be gained from an e-commerce platform, CRM and other market related external sources. 

The purpose of this project is to share with fellow sales, marketing professionals and data scientist how to approach quantitative part of customer journey mapping, namely answer questions:
 1. How many buyer personas do we have?
 2. What are their unique characteristics?
 3. How accurately can we predict buyer persona from the first customer purchase transaction?
 4. How can we adapt the marketing strategy concerning buyer personas to increase ROI?
 
 From a data science perspective it means:
 1. How to use hierarchical and non-hierarchical clustering to identify buyer personas?
 2. How to use ensemble and linear-based models to profile buyer personas characteristics?
 
## Results<a name="results"></a>

The main findings of the code can be found at the post available [here](TBD).

## Installation <a name="installation"></a>

There are several necessary 3rd party libraries beyond the Anaconda distribution of Python which needs to be installed and imported to run code. These are:
 - [scikit_posthocs](https://scikit-posthocs.readthedocs.io/en/latest/) providing posthoc tests for multiple comparison
 - [google cloud SDK](https://anaconda.org/conda-forge/google-cloud-sdk) providing access to BigQuerry and [Google Analytics Sample Dataset](https://support.google.com/analytics/answer/7586738?hl=en)
 
## File Descriptions <a name="files"></a>

There is 1 notebook available here to showcase work related to the above questions.  Markdown cells were used to assist in walking through the thought process for individual steps.  

There are additional files:
 - `bigquery_.py` provides custom classes of BigqueryTable and BiqqueryDataset to query data to [Google Merchandise Store](https://www.googlemerchandisestore.com/) sample dataset.
 - `helper_py` provides custom functions for various analyses, to keep notebook manageable to read. 
 - `custom_pca.py` holds adaptation of [scikit-learn PCA class](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html) including Varimax rotation and Latent Root criterion
 - `google_analytics_schema.xlsx` contains an analysis of variables in [Big Query Export Schema](https://support.google.com/analytics/answer/3437719?hl=en) used as a schema for Google Analytics Sample.
 - `product_categories.xlsx` ensures encoding of broken product category variables in the dataset
 - `temp.data.h5` stores codes/levels of each variable in the dataset
 
## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Must give credit to Google for the data.  @alexisbcook for a nice introduction to [Nested and Repeated Data](https://www.kaggle.com/alexisbcook/nested-and-repeated-data). Daqing Chen, Sai Laing Sain & Kun Guo for their technical article [Data mining for the online retail industry: A case study of RFM model-based customer segmentation using data mining](https://link.springer.com/article/10.1057/dbm.2012.17)
