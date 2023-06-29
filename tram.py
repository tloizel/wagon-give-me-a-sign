from load_from_bq import load_from_bq
from data_proc import preproc
from model import create_and_fit_model
from registry import save_model
from model import create_and_fit_model_ml, upload_model_ml#create_and_fit_model_merged, create_and_fit_model_merged_bi

df = load_from_bq()

X_train_df, X_test_df, y_train_df, y_test_df = preproc(df, test_size=0.3, random_state=42)


#model = create_and_fit_model_ml(X_train_df, y_train_df)


model = create_and_fit_model(X_train_df, y_train_df)
save_model(model, False, 'ML_test_gcs_1')

#model.save('models/LSTM_2')

#upload_model_ml(model, "random_forest_1")


"""
MERGED MODELS LAUNCHING
"""
#model_deep_merged = create_and_fit_model_merged(X_train_df, y_train_df)


#model_deep_merged_bi = create_and_fit_model_merged_bi(X_train_df, y_train_df)


#model_deep_merged.save('models/model_deep_merged')

#model_deep_merged_bi.save('models/model_deep_merged_bi')



#model = create_and_fit_model_ml(X_train_df, y_train_df)




"""
BOOST MODEL LAUNCHING
"""

model = create_and_fit_model_ml(X_train_df, X_test_df)

upload_model_ml(model, "boost")


"""
MERGED AND TRANSFORMER MODEL LAUNCHING
"""
