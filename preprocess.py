import cv2
import numpy as np
import os

# Les noms de vos dossiers
folders = ['raw_data/0', 'raw_data/1', 'raw_data/2']

# Taille de l'image pour le redimensionnement
image_size = (64, 64)

# Conteneurs pour les images et les étiquettes
images = []
labels = []

# Parcourez chaque dossier
for i, folder in enumerate(folders):
    # Parcourez chaque fichier dans le dossier
    for filename in os.listdir(folder):
        # Construire le chemin complet du fichier
        file = os.path.join(folder, filename)

        # Ouvrir l'image et la convertir en niveaux de gris
        image = cv2.imread(file, cv2.IMREAD_GRAYSCALE)

        # Redimensionner l'image
        image = cv2.resize(image, image_size)

        # Normaliser les valeurs de pixel
        image = image / 255.0

        # Ajouter l'image et l'étiquette à nos listes
        images.append(image)
        labels.append(i)

# Convertir les listes en tableaux numpy pour l'entraînement
images = np.array(images)
labels = np.array(labels)

print(images[0].shape)
