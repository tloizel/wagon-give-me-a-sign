import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

import numpy as np
# séparation des données


def create_and_fit_model_ml(X_train, y_train, timesteps=10):

    """
    Créer et entrainer le model de Machine learning (ici un LSTM)
    """

    #X_train = np.reshape(X_train, (X_train.shape[0], timesteps, X_train.shape[1] // timesteps))


    #early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    ## Convertir les données
    #X_train = X_train.values.reshape((X_train.shape[0], X_train.shape[1], 1)) # Reshape to (n_samples, n_timesteps, n_features)
#
    ## Encoder les labels
    #encoder = LabelEncoder()
    #y_train = encoder.fit_transform(y_train.squeeze()) # Convertir les lettres en chiffres
    #y_train = to_categorical(y_train) # Convertir en one-hot vectors
#
    ## Créer le modèle
    #model = Sequential()
    #model.add(LSTM(64, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)))
    #model.add(Dropout(0.5))
    #model.add(LSTM(64, activation='relu'))
    #model.add(Dropout(0.5))
    #model.add(Dense(y_train.shape[1], activation='softmax')) # y_train.shape[1] correspond au nombre de classes
#
    ## Compiler le modèle
    #model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#
    ## Entraîner le modèle
    #model.fit(X_train, y_train, epochs=100, validation_split=0.2, callbacks=[early_stopping])
    encoder= LabelEncoder()
    y_train = encoder.fit_transform(y_train)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    # définir le modèle
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(len(set(y_train)), activation='softmax'))

    # compiler le modèle
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # former le modèle
    model.fit(X_train, y_train, epochs=1000, batch_size=16, validation_split=0.30, callbacks=[early_stopping])
    return model



def predict_model_ml(model, X_to_predict):
    """
    Retourne une prédiction sur le model
    """
    return model.predict(X_to_predict)



def upload_model(model, name_of_model):
    with open(f'models/model.{name_of_model}', 'wb') as file:
        pickle.dump(model, file)


def load_model(name_of_the_model=False):
    if name_of_the_model == False:
        with open('models/model.randomfo', 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    else:
        with open(f'models/{name_of_the_model}', 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
