import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
import pickle


def create_and_fit_model_ml(X_train, y_train):

    """
    Créer et entrainer le model de Machine learning (ici un Random Forest Classifier)
    """

    # Créez une instance du classificateur
    clf = RandomForestClassifier()

    # Définissez la grille de paramètres que vous souhaitez explorer
    param_grid = {
        'n_estimators': [50, 100, 200,500, 1000],
        'max_depth': [None, 10, 20, 30, 50],
        'min_samples_split': [2, 5, 10, 30]
    }

    # Créez une instance de GridSearchCV
    grid_search = GridSearchCV(estimator=clf, param_grid=param_grid, cv=10, scoring='accuracy', verbose=3, n_jobs=-1)

    # Entraînez le GridSearchCV pour trouver les meilleurs paramètres
    grid_search.fit(X_train, y_train)

    # Affichez les meilleurs paramètres trouvés par la recherche sur grille
    print("Best parameters found: ", grid_search.best_params_)




    # Utilisez le meilleur modèle pour faire des prédictions
    best_clf = grid_search.best_estimator_

    return best_clf


def evaluate_model_ml_crossval(model, X, y, cv=5):
    """
    Evaluer le model de Machine learning avec méthode de cross vall et CV de 5
    """
    scores = cross_val_score(model, X, y, cv=cv)

    print("Scores de validation croisée : ", scores)
    print("Moyenne des scores de validation croisée : ", scores.mean())
    return scores.mean()

def evaluate_model_ml_accuracy(model, X_test, y_test):
    """
    Evalue l'accuracy du model
    """
    return accuracy_score(X_test, y_test)


def predict_model_ml(model, X_to_predict):
    """
    Retourne une prédiction sur le model
    """
    model.predict(X_to_predict)



def upload_model(model, name_of_model):
    with open(f'model.{name_of_model}', 'wb') as file:
        pickle.dump(model, file)


def load_model(name_of_the_model=False):
    if name_of_the_model == False:
        with open('model.randomfo', 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model
    else:
        with open(name_of_the_model, 'rb') as file:
            loaded_model = pickle.load(file)
        return loaded_model