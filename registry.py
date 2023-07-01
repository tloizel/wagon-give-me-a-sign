import glob
import os
import time
# import pickle

from colorama import Fore, Style

# added only for DL
# from tensorflow import keras

from params import *
import os
import joblib

# from google.oauth2 import service_account


def save_model(model, ml=False, model_name="no_name_model") -> None:
    """
    - Persist trained model locally on the hard drive at f"/models/model_name/{timestamp}.h5" for deep models or f"/models/model_name/{timestamp}.joblib for ML models"
    - if MODEL_TARGET='gcs', also persist it in the bucket on GCS at "models/model_name/{timestamp}.h5/joblib"
    """

    if ml is not True:
    # Save DEEP model locally
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        model_path = os.path.join("models", f"DL_{model_name}" ,f"{timestamp}.h5")
        model.save(model_path)

        print("✅ Deep Model saved locally")
    else:
    # Save ML model locally
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        directory = f'models/ML_{model_name}'
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
        client = storage.Client(project=PROJECT_ID, credentials=GCP_SERVICE_ACCOUNT)

        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(f"{model_path}")
        blob.upload_from_filename(model_path)

        print("✅ Model saved to GCS")

        return None

    return None





def load_model(ml=False, model_name="no_name_model", timestamp=""): #-> keras.Model:
    """
    Return a saved model:
    - locally (latest one in alphabetical order)
    - or from GCS (most recent one) if MODEL_TARGET=='gcs'
    - or from MLFLOW (by "stage") if MODEL_TARGET=='mlflow'

    Return None (but do not Raise) if no model is found
    """

    # if MODEL_TARGET == "local":
    #     print(Fore.BLUE + f"\nLoad latest model from local registry..." + Style.RESET_ALL)

    #     # Get the latest model version name by the timestamp on disk
    #     local_model_directory = os.path.join("models", "ML_*" if ml else "DL_*")
    #     local_model_paths = glob.glob(f"{local_model_directory}/*")

    #     if not local_model_paths:
    #         return None

    #     most_recent_model_path_on_disk = sorted(local_model_paths)[-1]

    #     print(Fore.BLUE + f"\nLoad latest model from disk..." + Style.RESET_ALL)

    #     if ml:
    #         latest_model = joblib.load(most_recent_model_path_on_disk)
    #     else:
    #         latest_model = keras.models.load_model(most_recent_model_path_on_disk)

    #     print("✅ Model loaded from local disk")

    #     return latest_model

    prefix = "DL" if not ml else "ML"

    if MODEL_TARGET == "gcs":
        from google.cloud import storage
        print(Fore.BLUE + f"\nLoad model {model_name} from GCS..." + Style.RESET_ALL)

        # credentials = service_account.Credentials.from_service_account_file("bq_keys.json")
        client = storage.Client(project=PROJECT_ID, credentials=GCP_SERVICE_ACCOUNT)
        blobs = list(client.get_bucket(BUCKET_NAME).list_blobs(prefix=f"models/{prefix}_{model_name}/{timestamp}"))

        if blobs:
            latest_blob = max(blobs, key=lambda x: x.updated)
            latest_model_path_to_save = os.path.join(latest_blob.name)

            if ml is not True:
                os.makedirs(os.path.dirname(latest_model_path_to_save), exist_ok=True)
                latest_blob.download_to_filename(latest_model_path_to_save)
                # latest_model = keras.models.load_model(latest_model_path_to_save)
                print(f"✅ Latest model {model_name} DEEP downloaded from cloud storage")
                return latest_model
            else:
                os.makedirs(os.path.dirname(latest_model_path_to_save), exist_ok=True)
                latest_blob.download_to_filename(latest_model_path_to_save)
                latest_model = joblib.load(latest_model_path_to_save)
                print(f"✅ Latest model {model_name} ML downloaded from cloud storage")
                return latest_model
        else:
            print(f"\n❌ No model {model_name} found in GCS bucket {BUCKET_NAME}")
            return None

if __name__ == "__main__":
    load_model()
