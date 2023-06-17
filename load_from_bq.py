import pandas
from google.cloud import bigquery
from google.oauth2 import service_account
from params import PROJECT_ID, TABLE_ID

def load_from_bq():
    credentials = service_account.Credentials.from_service_account_file("bq_keys.json")
    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
    table = client.get_table(TABLE_ID)

    # Construct the SQL query to retrieve the table data
    query = f'SELECT * FROM `{table}`'

    # Submit the query and fetch the results
    df = client.query(query).to_dataframe()
    return df

