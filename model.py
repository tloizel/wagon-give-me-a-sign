from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Conv1D, MaxPooling1D, Flatten, Bidirectional, RepeatVector, MultiHeadAttention, LayerNormalization, Input
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping
import pickle
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
import tensorflow as tf
from keras.layers import Concatenate
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from keras.layers import BatchNormalization, LeakyReLU
from sklearn import preprocessing


'''
PARTIE RESERVEE A LA FONCTION DE CREATION DE TRANSFORMER
'''

def transformer_block(embed_dim, num_heads, ff_dim, rate=0.1):
    """
    block de layer transformer
    """
    inputs = Input(shape=(None, embed_dim))
    attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)(inputs, inputs)
    attn_output = Dropout(rate)(attn_output)
    out1 = LayerNormalization(epsilon=1e-6)(inputs + attn_output)
    ffn_output = Dense(ff_dim, activation="relu")(out1)
    ffn_output = Dense(embed_dim)(ffn_output)
    ffn_output = Dropout(rate)(ffn_output)
    return Model(inputs=inputs, outputs=LayerNormalization(epsilon=1e-6)(out1 + ffn_output))



def create_and_fit_model_merged_transformer(X_train, y_train, timesteps=10):
    """
    merged model avec CNN et transformer
    """
    unique_labels, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} examples")

    le = LabelEncoder()
    num_classes = len(np.unique(y_train))

    early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
    n_features = X_train.shape[1]

    n_timesteps = 10
    n_samples_train = np.floor(X_train.shape[0] / n_timesteps).astype(int)
    X_train = np.resize(X_train, (n_samples_train*n_timesteps, n_features))
    X_train_lstm = X_train.reshape(n_samples_train, n_timesteps, n_features)

    le.fit(y_train)
    y_train_encoded = le.fit_transform(y_train.values.ravel())

    y_train_encoded = np.resize(y_train_encoded, (n_samples_train*n_timesteps,))
    y_train_encoded = to_categorical(y_train_encoded, num_classes)
    y_train_encoded = y_train_encoded.reshape(n_samples_train, n_timesteps, num_classes)

    # Transformer parameters
    embed_dim = n_features  # This must match the number of features
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    transformer = transformer_block(embed_dim, num_heads, ff_dim)

    # Modèle CNN
    cnn_model = Sequential()
    cnn_model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps, n_features)))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Conv1D(128, 3, activation='relu'))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Flatten())
    cnn_model.add(RepeatVector(n_timesteps))

    # Transformer input
    transformer_input = Input(shape=(n_timesteps, embed_dim))
    transformer_output = transformer(transformer_input)

    # Merge models
    merged_output = Concatenate()([cnn_model.output, transformer_output])

    # Add fully connected layer and output layer
    merged_output = Dense(128, activation='relu')(merged_output)
    output = Dense(num_classes, activation='softmax')(merged_output)

    # Create the model
    merged_model = Model(inputs=[cnn_model.input, transformer_input], outputs=output)

    optimizer = Adam(learning_rate=0.001)
    merged_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    merged_model.fit([X_train_lstm, X_train_lstm], y_train_encoded, validation_split=0.30, callbacks=[early_stopping], epochs=100, verbose=1, batch_size=16)

    return merged_model



def create_and_fit_model_merged_bi(X_train, y_train, timesteps=10):
    """
    merged model avec CNN et RNN bidirectionnel
    """
    unique_labels, counts = np.unique(y_train, return_counts=True)
    for label, count in zip(unique_labels, counts):
        print(f"Label {label}: {count} examples")

    le = LabelEncoder()
    num_classes = len(np.unique(y_train))

    early_stopping = EarlyStopping(monitor='val_loss', patience=35, restore_best_weights=True)
    n_features = X_train.shape[1]

    n_timesteps = 10
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
    cnn_model.add(Flatten())
    cnn_model.add(RepeatVector(n_timesteps))

    # Modèle Bi-LSTM
    lstm_model = Sequential()
    lstm_model.add(Bidirectional(LSTM(100, return_sequences=True), input_shape=(n_timesteps, n_features)))
    lstm_model.add(Dropout(0.2))

    # Merge models
    merged_output = Concatenate()([cnn_model.output, lstm_model.output])

    # Add fully connected layer and output layer
    merged_output = Dense(128, activation='relu')(merged_output)
    output = Dense(num_classes, activation='softmax')(merged_output)

    # Create the model
    merged_model = Model(inputs=[cnn_model.input, lstm_model.input], outputs=output)


    optimizer = Adam(learning_rate=0.001)
    merged_model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

    merged_model.fit([X_train_lstm, X_train_lstm], y_train_encoded, validation_split=0.30, callbacks=[early_stopping], epochs=100, verbose=1)

    return merged_model





def create_and_fit_model_merged(X_train, y_train, timesteps=10):
    """
    merged model avec CNN et RNN
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

    le.fit(y_train)
    y_train_encoded = le.fit_transform(y_train.values.ravel())

    y_train_encoded = np.resize(y_train_encoded, (n_samples_train*n_timesteps,))
    y_train_encoded = to_categorical(y_train_encoded, num_classes)
    y_train_encoded = y_train_encoded.reshape(n_samples_train, n_timesteps, num_classes)

    # Modèle CNN
    # Modèle CNN
    cnn_model = Sequential()
    cnn_model.add(Conv1D(64, 3, activation='relu', input_shape=(n_timesteps, n_features)))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Conv1D(128, 3, activation='relu'))
    cnn_model.add(MaxPooling1D(2))
    cnn_model.add(Flatten())
    cnn_model.add(RepeatVector(n_timesteps))

    # Modèle LSTM
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, input_shape=(n_timesteps, n_features), return_sequences=True))
    lstm_model.add(Dropout(0.2))

    # Merge models
    merged_output = Concatenate()([cnn_model.output, lstm_model.output])

    # Add fully connected layer and output layer
    merged_output = Dense(128, activation='relu')(merged_output)
    output = Dense(num_classes, activation='softmax')(merged_output)

    # Create the model
    merged_model = Model(inputs=[cnn_model.input, lstm_model.input], outputs=output)


    optimizer = Adam(learning_rate=0.001)
    merged_model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    merged_model.fit([X_train_lstm, X_train_lstm], y_train_encoded, validation_split=0.30, callbacks=[early_stopping], epochs=100, verbose=1)

    return merged_model


def create_and_fit_model_ml(X_train, y_train):
    """
    Model de ML_xgboost
    """
    y_train = np.ravel(y_train)

    le = preprocessing.LabelEncoder()

    # Fit the encoder to the pandas column
    le.fit(y_train)

    # Apply the fitted encoder to the pandas column
    y_train = le.transform(y_train)
    # Définir les paramètres pour la recherche de grille
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [2, 4, 6],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.5, 0.7, 1.0],
        'colsample_bytree': [0.4, 0.6, 0.8, 1.0],
    }

    # Créer un modèle XGBoost de base
    xgb_model = xgb.XGBClassifier()

    # Instancier la recherche de grille
    grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=10)

    # Ajuster la recherche de grille aux données
    grid_search.fit(X_train, y_train)

    # Le meilleur score trouvé par GridSearchCV
    best_accuracy = grid_search.best_score_

    # Les meilleurs paramètres trouvés par GridSearchCV
    best_params = grid_search.best_params_

    print(f"L'exactitude du meilleur modèle est: {best_accuracy}")
    print(f"Les meilleurs paramètres sont: {best_params}")

    return grid_search.best_estimator_


def create(X_train, y_train):
    """
    Model de base qui marche en random forest
    """
    #1 n_estimators=1000  min_samples_split=2
    # Définir les paramètres pour la recherche de grille
    clf = RandomForestClassifier(max_depth=None, min_samples_split=2, n_estimators=10, verbose=1, n_jobs=-1)
    clf.fit(X_train, y_train)

    # Afficher les meilleurs paramètres
    return clf





"""
PARTIE RESERVE AUX 'UTILS' DE MODELS, A SAVOIR LES FONCTIONS DE PREDICTION ET DE CHARGEMENT DE MODELS
"""

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
