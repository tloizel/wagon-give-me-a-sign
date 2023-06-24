import pandas
from google.cloud import bigquery
from google.oauth2 import service_account
from params import PROJECT_ID, TABLE_ID
from data_extraction import get_coordinates_from_collection
from utils import delete_local_enchantillon


def send_to_bq():
    """
        Envoi de notre échantillon de donnée sur Big Query et suppression de notre échantillon local
    """

    # Envoi de notre échantillon
    df = get_coordinates_from_collection()

    credentials = service_account.Credentials.from_service_account_file("bq_keys.json")

    client = bigquery.Client(project=PROJECT_ID, credentials=credentials)
    client.load_table_from_dataframe(df, TABLE_ID)

    # suppression de notre échantillon local selon volonté user
    question = input("Veux tu vraiment supprimer ta donnée stockée en local ? Y/n :        ")
    if question in ["Y", "y"]:
        delete_local_enchantillon()
        print("Echantillon local bien supprimé et donnée bien envoyée sur Big Query")
    else:
        print("Donnée bien envoyée sur Big Query")

    pass

if __name__ == "__main__":
    send_to_bq()
