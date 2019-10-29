"""Bigquery Custom Module


Module serves for custome classes and functions to access Google's bigquery
"""

# IMPORTS
# -------

# internal constants
__ENV_VAR_NAME__ = "GOOGLE_APPLICATION_CREDENTIALS"


# Standard libraries
import os
import numpy as np
import pandas as pd
import ipdb

# 3rd party libraries
from google.cloud import bigquery
from tqdm import tqdm_notebook as tqdm

def authenticate_service_account(service_account_key):
    """Set Windows environment variable for bigquery authentification

    Args:
        service_account_key: path to service account key
    """
    if not __ENV_VAR_NAME__ in os.environ:
        os.environ[__ENV_VAR_NAME__] = service_account_key
    else:
        raise Exception(
            'Environment variable {} = {} already exists.'
            .format(__ENV_VAR_NAME__, os.environ[__ENV_VAR_NAME__])
        )


class BigqueryTable (bigquery.table.Table):
    '''Bigquery table object customized to work with Pandas'''

    def _append_schema_field(
        self, schema_list, schema_field, parent_field_name=''):
        '''Explodes schema field and append it to list of exploded schema fields

        Args:
            schema_list: list of schema fields
            schema_field: bigquery.schema.SchemaField object
            parent_field_name: field name of patent schema, None by defualt

        Returns:
            schema_list: updated schema list with appened schema_field record
        '''

        # explode schema field
        schema_field = (schema_field.name if parent_field_name == '' \
                        else parent_field_name + '.' + schema_field.name,
                        schema_field.field_type,
                        schema_field.mode,
                        schema_field.description)

        # append exploded schema to list of exploded schemas
        schema_list.append(schema_field)
        return schema_list

    def _traverse_schema(self, schema_list, schema_field,
                        parent_field_name = ''):
        '''Traverses table schema, unnest schema field objects, explodes their
        attributes to list of tuples

        Args:
            schema_list: unnnested and exploded schema list
            schema_field: bigquery.schema.SchemaField object
            parent_field_name: field name of patent schema, empty string by default

        Returns:
            schema_list: updated schema list by schema_field
        '''

        # for nested schema field go deeper
        if schema_field.field_type == 'RECORD':

            # explode and append parent schema field to schema list
            schema_list = self._append_schema_field(
                schema_list, schema_field, parent_field_name)

            # explode and append children schema fields
            for child_field in schema_field.fields:
                schema_list = self._traverse_schema(
                    schema_list, child_field,
                    schema_field.name if parent_field_name == ''
                    else parent_field_name + '.' + schema_field.name
                )

        # explode and append not nested schema field
        else:
            schema_list = self._append_schema_field(
                schema_list, schema_field, parent_field_name)

        return schema_list

    def schema_to_dataframe(self, bq_exp_schema=None):
        '''Unnest schema and recast it to dataframe

        Args:
            bq_exp_schema: BigQuery export schema as dataframe with descriptions,
                           None by default

        Returns:
            schema_df: Table schema as dataframe
        '''

        # traverse and explode table schema to list of tuples
        schema_list = []
        for schema_field in self.schema:
            schema_list = self._traverse_schema(schema_list, schema_field)

        # transform exploded schema to dataframe
        schema_df =  pd.DataFrame(
            np.array(schema_list),
            columns=['Field Name', 'Data Type', 'Mode', 'Description']
        ).set_index('Field Name')

        # merge BigQuery export schema with table schema to get field
        # description
        if bq_exp_schema is not None:

            # merge schemas
            bq_exp_schema = bq_exp_schema.set_index('Field Name')
            schema_df = (
                pd.merge(schema_df.drop(columns=['Description']),
                         bq_exp_schema['Description'],
                         left_index=True, right_index=True,
                         how='left')
                .fillna('Not specified in BigQuerry Export Schema')
            )

            # match depreciated fields
            depreciated_fields = {
                'hits.appInfo.name': 'hits.appInfo.appName',
                 'hits.appInfo.version': 'hits.appInfo.appVersion',
                 'hits.appInfo.id': 'hits.appInfo.appId',
                 'hits.appInfo.installerId': 'hits.appInfo.appInstallerId',

                 'hits.publisher.adxClicks':
                 'hits.publisher.adxBackfillDfpClicks',
                 'hits.publisher.adxImpressions':
                 'hits.publisher.adxBackfillDfpImpressions',
                 'hits.publisher.adxMatchedQueries':
                 'hits.publisher.adxBackfillDfpMatchedQueries',
                 'hits.publisher.adxMeasurableImpressions':
                 'hits.publisher.adxBackfillDfpMeasurableImpressions',
                 'hits.publisher.adxQueries':
                 'hits.publisher.adxBackfillDfpQueries',
                 'hits.publisher.adxViewableImpressions':
                 'hits.publisher.adxBackfillDfpViewableImpressions',
                 'hits.publisher.adxPagesViewed':
                 'hits.publisher.adxBackfillDfpPagesViewed'
            }
            for ga_field, exp_field in depreciated_fields.items():
                schema_df['Description'][ga_field] = (
                    bq_exp_schema['Description'][exp_field])

            # match multiple fields
            multiple_fields = {
                'totals.uniqueScreenviews': 'totals.UniqueScreenViews',
                'hits.contentGroup.contentGroup':
                'hits.contentGroup.contentGroupX',
                'hits.contentGroup.previousContentGroup':
                'hits.contentGroup.previousContentGroupX',
                'hits.contentGroup.contentGroupUniqueViews':
                'hits.contentGroup.contentGroupUniqueViewsX'}

            fields = schema_df.index
            for ga_field, exp_field in multiple_fields.items():
                schema_df['Description'][fields.str.contains(ga_field)] = (
                    bq_exp_schema['Description'][exp_field]
                )

        return schema_df.reset_index()

    def display_schema(self, schema_df):
        '''Displays left justified schema to full cell width

        Args:
            schema_df: Exploded bigquerry schema as dataframe
        '''
        with pd.option_context('display.max_colwidth', 999):
            display(
                schema_df.style.set_table_styles(
                    [dict(selector='th', props=[('text-align', 'left')]),
                     dict(selector='td', props=[('text-align', 'left')])])
            )


    def to_dataframe(self, client):
        '''Expands and Converts table to DataFrame

        Args:
            client: instatiated Bigquery client

        Returns:
            df: Dataframe with expanded table fields
        '''

        # convert table schema to DataFrame
        schema = self.schema_to_dataframe(bq_exp_schema=None)

        # extract SELECT aliases from schema
        repeated_fields = schema[schema['Mode'] == 'REPEATED']
        select_aliases = schema.apply(self._get_select_aliases,
                                      axis=1,
                                      args=(repeated_fields,))

        select_aliases = select_aliases[select_aliases['Data Type'] != 'RECORD']
        select = ',\n'.join(
            (select_aliases['Field Name']
             + ' AS '
             + select_aliases['Field Alias'])
            .to_list()
        )

        # extract FROM aliases from schema
        from_aliases = repeated_fields.apply(self._get_from_aliases,
                                             axis=1,
                                             args=(repeated_fields,))
        from_ = '\n'.join(
            ('LEFT JOIN UNNEST(' + from_aliases['Field Name']
             + ') AS '
             + from_aliases['Field Alias'])
            .to_list()
        )

        # query table and return as dataframe
        query = '''
            SELECT
                {}
            FROM
                {}
                {}
        '''.format(select,
                   '`' + self.full_table_id.replace(':', '.') + '`',
                   from_)
        df = client.query(query).to_dataframe()
        return df

    def _get_select_aliases(self, field, repeated_fields):
        '''Create aliases for nested fields for SELECT statment'''

        if '.' in field['Field Name'] and field['Mode'] != 'REPEATED':
            parent = '.'.join(field['Field Name'].split('.')[:-1])
            child = field['Field Name'].split('.')[-1]

            if parent in repeated_fields.values:
                field['Field Name'] = parent.replace('.', '_') + '.' + child

        field['Field Alias'] = field['Field Name'].replace('.', '_')

        return field

    def _get_from_aliases(self, field, repeated_fields):
        '''Create aliases for nested fields for FROM statment'''

        if '.' in field['Field Name']:
            parent = '.'.join(field['Field Name'].split('.')[:-1])
            child = field['Field Name'].split('.')[-1]

            if '.' in parent:
                field['Field Name'] = parent.replace('.', '_') + '.' + child

        field['Field Alias'] = field['Field Name'].replace('.', '_')

        return field


class BigqueryDataset(bigquery.dataset.Dataset):
    '''Customize biqquery Dataset Class'''

    def __init__(self, client, *args, **kwg):
        super().__init__(*args, **kwg)
        self.schema = self.get_information_schema(client)

    def get_information_schema(self, client):
        '''Returns information schema including table names in dataset
           as dataframe

           Args:
               client: gigquery.client.Client object

           Returns:
               schema_df: dataset information schema as DataFrame
        '''

        query = '''
            SELECT
             * EXCEPT(is_typed)
            FROM
             `{project}.{dataset_id}`.INFORMATION_SCHEMA.TABLES
        '''.format(project=self.project, dataset_id=self.dataset_id)
        schema = (client.query(query)
                        .to_dataframe()
                        .sort_values('table_name'))
        return schema

    def get_levels(self, client, bq_exp_schema=None):
        '''Add to table schema two columns: 'Levels' which contains
        unique values of each variable as set and 'Num of Levels'
        determining number of unique values. Purpose is to get
        fealing what values are hold by variable and scale of the variable
        (Ex: Unary, Binary, Multilevel etc...)

        Note: Functions may run long time for big datasets as sequentialy
        loads all tables in dataset to safe memory.

        Args:
            client: bigquery.client.Client object
            bq_query_exp_schema: BigQuery export schema including field names
                                 Description as dataframe
        Returns:
            schema: updated schema with level charateristics as
                    dataframe
        '''

        # union variable levels of consecutive queried tables
        def level_union(var, var_levels):
            try:
                var['Levels'] = var['Levels'] | var_levels[var.name]
            except:
                pass
            return var['Levels']

        # get last table of dataset
        dataset_ref = client.dataset(self.dataset_id, self.project)
        table_id = self.schema['table_name'].values[-1]
        table_ref = dataset_ref.table(table_id)
        table = client.get_table(table_ref)
        table.__class__ = BigqueryTable

        # get last table schema
        schema = table.schema_to_dataframe(bq_exp_schema)

        # keep only non record fields in schema
        schema = schema[schema['Data Type'] != 'RECORD'].copy()

        # ad variable names
        var_name = (schema['Field Name']
                    .apply(lambda field: field.replace('.', '_')))
        schema.insert(0, 'Variable Name', var_name)
        schema.insert(4, 'Levels', [set()] * schema.index.size)
        schema = schema.set_index('Variable Name')

        # analyze variable codes/levels
        for i, table_id in tqdm(enumerate(self.schema['table_name'])):

            # load table
            table_ref = dataset_ref.table(table_id)
            table = client.get_table(table_ref)
            table.__class__ = BigqueryTable
            df = table.to_dataframe(client)

            # get and update variable levels
            var_levels = df.apply(lambda var: set(var.unique()))
            schema['Levels'] = (
                schema.apply(level_union, args=(var_levels,),
                              axis = 1)
            )

            if i == 2: break

        # calculate number of variable levels
        schema.insert(4, 'Num of Levels',
                       schema.apply(lambda var: len(var['Levels']),
                                     axis=1)
                      )
        return schema
