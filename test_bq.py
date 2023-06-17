import pandas
from google.cloud import bigquery
from google.oauth2 import service_account
from params import PROJECT_ID, TABLE_ID

from data_extraction import get_coordinates

df = get_coordinates()

credentials = service_account.Credentials.from_service_account_file("bq_keys.json")

client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
client.load_table_from_dataframe(df, TABLE_ID)

# print(client.get_table(table_id).num_rows)
