import pandas
import pandas_gbq
from google.cloud import bigquery
from google.oauth2 import service_account

credentials = service_account.Credentials.from_service_account_file("bg_keys.json")


# TODO: Set project_id to your Google Cloud Platform project ID.
project_id = "wagon-give-me-a-sign"

# TODO: Set table_id to the full destination table ID (including the
#       dataset ID).
table_id = 'test_dataset.test2'

df = pandas.DataFrame(
    {
        "letter": ["t", "b", "c"],
        "num": [4.0, 5.0, 6.0],
    }
)

client = bigquery.Client(project='wagon-give-me-a-sign', credentials=credentials)
client.load_table_from_dataframe(df, table_id)

# print(client.get_table(table_id).num_rows)
