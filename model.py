import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, GlobalMaxPooling1D, Reshape, RepeatVector,Lambda,TimeDistributed
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import tensorflow as tf
from keras.layers import Concatenate

import numpy as np
from keras.layers import BatchNormalization, LeakyReLU

def create_and_fit_model_merged_bi(X_train, y_train, timesteps=10):

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


    n_timesteps = 30
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




    # Modèle CNN
    cnn_model = Sequential()
    cnn_model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps, n_features)))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Conv1D(128, 3, activation='relu'))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Flatten())
    cnn_model.add(Dense(128, activation='relu'))

    bi_lstm_model = Sequential()
    bi_lstm_model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(n_timesteps, n_features)))
    bi_lstm_model.add(Dropout(0.2))
    bi_lstm_model.add(Dense(100, activation='relu'))
    bi_lstm_model.add(Dropout(0.2))

    # Fusion des modèles
    merged_model = Sequential()
    merged_model.add(tf.keras.layers.Concatenate([cnn_model, bi_lstm_model]))
    merged_model.add(Dense(num_classes, activation='softmax'))

    # Compilation du modèle
    optimizer = Adam(learning_rate=0.001)
    merged_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    # Entraînement du modèle
    merged_model.fit(X_train, y_train, validation_split=0.30, callbacks=[early_stopping], epochs=100, verbose=1)

    return merged_model



from keras.layers import Concatenate, Conv1D, MaxPooling1D, Flatten, Dense, LSTM, Dropout
from keras.models import Sequential
from keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam

def create_and_fit_model_merged(X_train, y_train, timesteps=10):
    unique_labels, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} examples")

    le = LabelEncoder()
    num_classes = len(np.unique(y_train))

    early_stopping = EarlyStopping(monitor='val_loss', patience=15)
    n_features = X_train.shape[1]

    n_timesteps = 30
    n_samples_train = np.floor(X_train.shape[0] / n_timesteps).astype(int)
    X_train = np.resize(X_train, (n_samples_train*n_timesteps, n_features))
    X_train_lstm = X_train.reshape(n_samples_train, n_timesteps, n_features)

    le.fit(y_train)
    y_train_encoded = le.fit_transform(y_train.values.ravel())

    y_train_encoded = np.resize(y_train_encoded, (n_samples_train*n_timesteps,))
    y_train_encoded = to_categorical(y_train_encoded, num_classes)
    y_train_encoded = y_train_encoded.reshape(n_samples_train, n_timesteps, num_classes)

    # Modèle CNN
    cnn_model = Sequential()
    cnn_model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps, n_features)))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Conv1D(128, 3, activation='relu'))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Conv1D(100, 3, activation='relu', padding='same'))
    cnn_model.add(MaxPooling1D(5)) # add another pooling layer to match the LSTM output shape


    # Modèle LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, input_shape=(n_timesteps, n_features), return_sequences=True)) # return sequences
    lstm_model.add(Dropout(0.2))

    # Merge models
    merged_output = Concatenate()([cnn_model.output, lstm_model.output])

    # Add fully connected layer and output layer
    merged_output = TimeDistributed(Dense(128, activation='relu'))(merged_output) # Add TimeDistributed wrapper
    output = TimeDistributed(Dense(num_classes, activation='softmax'))(merged_output) # Add TimeDistributed wrapper

    merged_model = Model(inputs=[cnn_model.input, lstm_model.input], outputs=output)

    optimizer = Adam(learning_rate=0.001)
    merged_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    merged_model.fit([X_train_lstm, X_train_lstm], y_train_encoded, validation_split=0.30, callbacks=[early_stopping], epochs=100, verbose=1)

    return merged_model



def create_and_fit_model(X_train, y_train, timesteps=10):

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
    n_timesteps = 1
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


def create_and_fit_model_ml(X_train, y_train, timesteps=10):

    import pandas as pd
    from sklearn.model_selection import GridSearchCV
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    # Créez une instance du classificateur
    clf = RandomForestClassifier()

    # Définissez la grille de paramètres que vous souhaitez explorer

    # Créez une instance de GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid={'max_depth': None, 'min_samples_split': 2, 'n_estimators': 1000}, cv=10, scoring='accuracy', verbose=10, n_jobs=-1)

    # Entraînez le GridSearchCV pour trouver les meilleurs paramètres
    grid_search.fit(X_train, y_train.values.ravel())

    # Affichez les meilleurs paramètres trouvés par la recherche sur grille
    print("Best parameters found: ", grid_search.best_params_)

    # Utilisez le meilleur modèle pour faire des prédictions
    best_clf = grid_search.best_estimator_

    return best_clf






def predict_model_ml(model, X_to_predict):
    """
    Retourne une prédiction sur le model
    """
    return model.predict(X_to_predict)



def upload_model_ml(model, name_of_model):
    with open(f'models/model.{name_of_model}', 'wb') as file:
        pickle.dump(model, file)


def load_model_ml(name_of_the_model=False):
    if name_of_the_model == False:
        with open('models/model.randomfo', 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    else:
        with open(f'models/{name_of_the_model}', 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
