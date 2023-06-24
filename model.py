import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import numpy as np
from keras.layers import BatchNormalization, LeakyReLU



def create_and_fit_model_ml(X_train, y_train, timesteps=10):

    """
    Créer et entrainer le model de Machine learning (ici un LSTM)
    """
    unique_labels, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} examples")


    le = LabelEncoder()
    num_classes = len(np.unique(y_train))

    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    n_features = X_train.shape[1]



    # Transformer les données pour qu'elles aient la bonne forme pour le LSTM

    # Reshaping X_train for LSTM
    n_timesteps = 7
    n_samples_train = np.floor(X_train.shape[0] / n_timesteps).astype(int)
    X_train = np.resize(X_train, (n_samples_train*n_timesteps, n_features))
    X_train_lstm = X_train.reshape(n_samples_train, n_timesteps, n_features)

    # Reshaping y_train to match X_train
# Fit the label encoder on the y_train data
    le.fit(y_train)
    y_train_encoded = le.fit_transform(y_train.values.ravel())

    # Transform y_train to corresponding encoded labels

    # Reshape y_train_encoded to match the (samples, timesteps) structure
    y_train_encoded = np.resize(y_train_encoded, (n_samples_train*n_timesteps,))

    # Convert the encoded y_train data to categorical format
    y_train_encoded = to_categorical(y_train_encoded, num_classes)

    # Finally, reshape y_train_encoded to be 3D to match X_train_lstm
    # The last dimension should be the number of classes
    y_train_encoded = y_train_encoded.reshape(n_samples_train, n_timesteps, num_classes)

    #n_samples_test = np.floor(X_train.shape[0] / n_timesteps).astype(int)
    y_train_labels = np.argmax(y_train_encoded, axis=1)





    model = Sequential()
    model.add(LSTM(128, input_shape=(n_timesteps, n_features), return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(128,  activation='relu'))
    model.add(LeakyReLU(alpha=0.05))
    model.add(Dense(64,  activation='relu'))
    model.add(Dropout(0.3))
    model.add(BatchNormalization())

    model.add(Dense(num_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    model.fit(X_train_lstm, y_train_encoded , validation_split=0.30, callbacks=[early_stopping], epochs=100, verbose=1)

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
