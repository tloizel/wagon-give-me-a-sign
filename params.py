import os
import numpy as np
import streamlit as st
from google.oauth2 import service_account

###### VARIABLES
# PROJECT_ID = os.environ.get("PROJECT_ID")
# TABLE_ID = os.environ.get("TABLE_ID")

# MODEL_TARGET = os.environ.get("MODEL_TARGET")
# BUCKET_NAME = os.environ.get("BUCKET_NAME")

PROJECT_ID = st.secrets["PROJECT_ID"]
TABLE_ID = st.secrets["TABLE_ID"]
MODEL_TARGET = st.secrets["MODEL_TARGET"]
BUCKET_NAME = st.secrets["BUCKET_NAME"]
TWILIO_ACCOUNT_SID = st.secrets['TWILIO_ACCOUNT_SID']
TWILIO_AUTH_TOKEN = st.secrets['TWILIO_AUTH_TOKEN']


GCP_SERVICE_ACCOUNT = service_account.Credentials.from_service_account_info(st.secrets['connections']['gcs'])
