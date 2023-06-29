from load_from_bq import load_from_bq
from data_proc import preproc
from registry import save_model, load_model
from model import load_model_ml #create_and_fit_model_merged, create_and_fit_model_merged_bi
#from model import create_and_fit_model_ml, upload_model_ml, create_and_fit_model_merged_transformer, load_model_ml
from sklearn.ensemble import RandomForestClassifier

import tensorflow as tf
# Créez une instance du classificateur


#df = load_from_bq()
#
#X_train_df, X_test_df, y_train_df, y_test_df = preproc(df, test_size=0.3, random_state=42)
#
#model = create_and_fit_model(X_train_df, y_train_df)
#
#save_model(model, False, 'ML_test_gcs_2')


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

#model = create_and_fit_model_ml(X_train_df, y_train_df)

#upload_model_ml(model, "boost")


"""
CNN AND TRANSFORMER MERGED MODEL LAUNCHING
"""

#model_de_la_muerte = create_and_fit_model_merged_transformer(X_train_df, y_train_df)
#model_de_la_muerte.save('models/model_de_la_muerte')



"""
CLF DE BASE
"""

#clf = create(X_train_df, y_train_df)

#upload_model_ml(clf, "model_base_testing")


#

"""
SAVE YOUR MODEL TO BQ
"""
#model = load_model('model.random_forest_1')



#model = tf.keras.models.load_model('models/model_de_la_muerte')
#save_model(model, ml=False, model_name="merged_cnn_transformer_1")


#dl_model = load_model(ml=False, model_name="merged_cnn_transformer_1")

ml_model = load_model(ml=True, model_name="random_forest_1", timestamp="20230629-193635")
print(ml_model)
