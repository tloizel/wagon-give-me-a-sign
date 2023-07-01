import os
import numpy as np
import streamlit as st

###### VARIABLES
# PROJECT_ID = os.environ.get("PROJECT_ID")
# TABLE_ID = os.environ.get("TABLE_ID")

# MODEL_TARGET = os.environ.get("MODEL_TARGET")
# BUCKET_NAME = os.environ.get("BUCKET_NAME")

PROJECT_ID = st.secrets["PROJECT_ID"]
TABLE_ID = st.secrets["TABLE_ID"]
MODEL_TARGET = st.secrets["MODEL_TARGET"]
BUCKET_NAME = st.secrets["BUCKET_NAME"]
