from load_from_bq import load_from_bq
from data_proc import preproc
from model import create_and_fit_model_ml
from registry import save_model

df = load_from_bq()

X_train_df, X_test_df, y_train_df, y_test_df = preproc(df, test_size=0.3, random_state=42)

model = create_and_fit_model_ml(X_train_df, y_train_df)

save_model(model, True, 'ML_test_gcs_1')
