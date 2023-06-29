import glob
import os
import time
import pickle

from colorama import Fore, Style
from tensorflow import keras

from params import *
import os
import joblib

from google.oauth2 import service_account

def save_model(model, ml=False, model_name="no_name_model") -> None:
    """
    - Persist trained model locally on the hard drive at f"/models/model_name/{timestamp}.h5" for deep models or f"/models/model_name/{timestamp}.joblib for ML models"
    - if MODEL_TARGET='gcs', also persist it in the bucket on GCS at "models/model_name/{timestamp}.h5/joblib"
    """
    if ml is not True:
    # Save DEEP model locally
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join("models", model_name ,f"{timestamp}.h5")
        model.save(model_path)

        print("✅ Deep Model saved locally")
    else:
    # Save ML model locally
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        directory = f'models/{model_name}'
        os.makedirs(directory, exist_ok=True)
        model_path = os.path.join(directory, timestamp + '.joblib')

        joblib.dump(model, model_path)
        # with open(f'{directory}/{timestamp}', 'wb') as file:
        #     pickle.dump(model, file)
        print("✅ ML Model saved locally")
        #model_path = os.path.join(f"{directory}",f"{timestamp}")


    if MODEL_TARGET == "gcs":
        from google.cloud import storage
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        # Update the service account email and project ID
        credentials = service_account.Credentials.from_service_account_file("bq_keys.json")
        client = storage.Client(project=PROJECT_ID, credentials=credentials)
        print(client)
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{model_path}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None

def load_model(ml=False) -> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'  --> for unit 02 only
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow' --> for unit 03 only

    Return None (but do not Raise) if no model is found

    """

    # if MODEL_TARGET == "local":
    #     print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    #     # Get the latest model version name by the timestamp on disk
    #     local_model_directory = os.path.join(LOCAL_REGISTRY_PATH, "models")
    #     local_model_paths = glob.glob(f"{local_model_directory}/*")

    #     if not local_model_paths:
    #         return None

    #     most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    #     print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    #     latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    #     print("✅ Model loaded from local disk")

    #     return latest_model

    if MODEL_TARGET == "gcs":
        from google.cloud import storage
        # 🎁 We give you this piece of code as a gift. Please read it carefully! Add a breakpoint if needed!
        print(Fore.BLUE + f"\nLoad latest model from GCS..." + Style.RESET_ALL)

        credentials = service_account.Credentials.from_service_account_file("bq_keys.json")
        client = storage.Client(project=PROJECT_ID, credentials=credentials)
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs())
        latest_blob = max(blobs, key=lambda x: x.updated)
        latest_model_path_to_save = os.path.join("models", "load_from_bq", latest_blob.name)

        if ml is not True:
        # Fonctionne pour deep
            os.makedirs(os.path.dirname(latest_model_path_to_save), exist_ok=True)
            latest_blob.download_to_filename(latest_model_path_to_save)
            latest_model = keras.models.load_model(latest_model_path_to_save)
            print("✅ Latest model DEEP downloaded from cloud storage")
            return latest_model
        else:
            #TO DO : mettre un model de ML en dernier sur le bucket pour voir si ca fontionne
            #Actualiser le code pour choisir le model qu'on veut charger
            latest_blob.download_to_filename(latest_model_path_to_save)
            latest_model = joblib.load(latest_model_path_to_save)
            print("✅ Latest model ML downloaded from cloud storage")
            return latest_model

        # try:



        # except:
        #     print(f"\n❌ No model found in GCS bucket {BUCKET_NAME}")

        #     return None


if __name__ == "__main__":
        load_model()
