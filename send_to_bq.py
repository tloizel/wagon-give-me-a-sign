import pandas
from google.cloud import bigquery
from google.oauth2 import service_account
from params import PROJECT_ID, TABLE_ID
from data_extraction import get_coordinates
import os
import shutil
import glob

DATA_DIR = './raw_data'

def send_to_bq():
    """
        Envoi de notre échantillon de donnée sur Big Query et suppression de notre échantillon local
    """

    # Envoi de notre échantillon
    df = get_coordinates()

    credentials = service_account.Credentials.from_service_account_file("bq_keys.json")

    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
    client.load_table_from_dataframe(df, TABLE_ID)


    # Suppression de notre échantillon local
    files = glob.glob(DATA_DIR)
    for f in files:
        if os.path.isfile(f):
            os.remove(f)
        elif os.path.isdir(f):
            shutil.rmtree(f)

    pass
