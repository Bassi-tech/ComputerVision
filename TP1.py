#%%
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import cv2

X = np.random.rand(1000)
X = 3 * X
moyenne = round(np.mean(X), 2)
ecart_type = round(np.std(X), 2)
median = round(np.median(X), 2)
print("Moyenne :", moyenne)
print("Écart-type :", ecart_type)
print("Médiane :", median)

# Fixez la graine aléatoire pour la reproductibilité des résultats
np.random.seed(42)

# Créez une liste X_bis de 1000 points avec des valeurs aléatoires dans [0, 3]
X_bis = 3 * np.random.rand(1000)
moyenne_bis = round(np.mean(X_bis), 2)
ecart_type_bis = round(np.std(X_bis), 2)
median_bis = round(np.median(X_bis), 2)

print("Résultats de X:")
print("Moyenne :", moyenne)
print("Écart-type :", ecart_type)
print("Médiane :", median)

print("\nRésultats de X_bis:")
print("Moyenne de X_bis :", moyenne_bis)
print("Écart-type de X_bis :", ecart_type_bis)
print("Médiane de X_bis :", median_bis)
# %%
sin_X = np.sin(X)
noise = 0.1 * np.random.randn(1000)
y = sin_X + noise

# %%
plt.figure(figsize=(8,6))
plt.scatter(X, y)
plt.show()
# %%
plt.figure(figsize=(8,6))
plt.hist(noise, bins=50, density=True, alpha=0.7, color='b', edgecolor='black')
plt.xlabel('Valeur du bruit')
plt.ylabel('Fréquence')
plt.title('Histogramme du bruit gaussien')
plt.grid(True)
plt.show()
# %%
import os
import glob

# Spécifiez le chemin du dossier que vous souhaitez explorer
dossier = os.path.expanduser(r'C:\Users\hbass\OneDrive\Bureau\ComputerVision\computer_vision_tp1\data1\bike')

# Spécifiez les extensions d'images que vous souhaitez compter
extensions_images = ['*.jpg', '*.jpeg', '*.png', '*.gif']

# Initialiser un compteur d'images
nombre_images = 0

# Parcourez les extensions d'images spécifiées
for extension in extensions_images:
    # Utilisez la bibliothèque glob pour trouver tous les fichiers avec cette extension dans le dossier
    fichiers_images = glob.glob(os.path.join(dossier, extension))
    # Incrémentez le compteur avec le nombre de fichiers trouvés
    nombre_images += len(fichiers_images)
print(f"Nombre total d'images dans le dossier : {nombre_images}")
dossier = os.path.expanduser(r'C:\Users\hbass\OneDrive\Bureau\ComputerVision\computer_vision_tp1\data1\car')

# Spécifiez les extensions d'images que vous souhaitez compter
extensions_images = ['*.jpg', '*.jpeg', '*.png', '*.gif']

# Parcourez les extensions d'images spécifiées
for extension in extensions_images:
    # Utilisez la bibliothèque glob pour trouver tous les fichiers avec cette extension dans le dossier
    fichiers_images = glob.glob(os.path.join(dossier, extension))
    # Incrémentez le compteur avec le nombre de fichiers trouvés
    nombre_images += len(fichiers_images)

# Affichez le nombre total d'images trouvées
print(f"Nombre total d'images dans le dossier : {nombre_images}")

# %%
import os
import glob
from PIL import Image

dossier1 = r'C:\Users\hbass\OneDrive\Bureau\ComputerVision\computer_vision_tp1\data1\bike'
dossier2 = r'C:\Users\hbass\OneDrive\Bureau\ComputerVision\computer_vision_tp1\data1\car'  # Spécifiez le chemin du deuxième dossier

# Spécifiez les extensions d'images que vous souhaitez inspecter
extensions_images = ['*.jpg', '*.jpeg', '*.png', '*.gif']

# Parcourez les extensions d'images spécifiées
for extension in extensions_images:
    # Utilisez la bibliothèque glob pour trouver tous les fichiers avec cette extension dans le premier dossier
    fichiers_images_dossier1 = glob.glob(os.path.join(dossier1, extension))
    
    # Utilisez la bibliothèque glob pour trouver tous les fichiers avec cette extension dans le deuxième dossier
    fichiers_images_dossier2 = glob.glob(os.path.join(dossier2, extension))
    
    # Parcourez chaque fichier image trouvé dans le premier dossier
    for fichier_image in fichiers_images_dossier1:
        # Ouvrez l'image avec PIL
        img = Image.open(fichier_image)
        
        # Obtenez le format (extension) de l'image
        format_image = img.format
        
        # Obtenez la taille (largeur x hauteur) de l'image
        taille_image = img.size
        
        # Affichez le format et la taille de l'image
        print(f"Dossier 1 - Image : {fichier_image}")
        print(f"Format : {format_image}")
        print(f"Taille : {taille_image}")
        print("\n")
    
    # Parcourez chaque fichier image trouvé dans le deuxième dossier
    for fichier_image in fichiers_images_dossier2:
        # Ouvrez l'image avec PIL
        img = Image.open(fichier_image)
        
        # Obtenez le format (extension) de l'image
        format_image = img.format
        
        # Obtenez la taille (largeur x hauteur) de l'image
        taille_image = img.size
        
        # Affichez le format et la taille de l'image
        print(f"Dossier 2 - Image : {fichier_image}")
        print(f"Format : {format_image}")
        print(f"Taille : {taille_image}")
        print("\n")

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Charger l'image en couleur
image_path = r'C:\Users\hbass\OneDrive\Bureau\ComputerVision\computer_vision_tp1\data1\bike\Bike (35).jpeg'
image = mpimg.imread(image_path)

# Afficher l'image en couleur
plt.imshow(image)
plt.title('Image en couleur')
plt.show()
# %%
image_bw = image[:, :, 1]

# Afficher l'image en noir et blanc
plt.imshow(image_bw, cmap='gray')
plt.title('Image en noir et blanc')
plt.show()

# %%
image = mpimg.imread(image_path)

# Convertir l'image en noir et blanc en utilisant uniquement le canal vert
image_bw = image[:, :, 1]

# Inverser l'image en noir et blanc en utilisant numpy.flipud
image_bw_inversed = np.flipud(image_bw)

# Afficher l'image en noir et blanc inversée
plt.imshow(image_bw_inversed, cmap='gray')
plt.title('Image en noir et blanc à l\'envers')
plt.show()
# %%
# Définir les chemins relatifs vers les dossiers "bike" et "car"
bike_folder = r'C:\Users\hbass\OneDrive\Bureau\ComputerVision\computer_vision_tp1\data1\bike'
car_folder = r'C:\Users\hbass\OneDrive\Bureau\ComputerVision\computer_vision_tp1\data1\car'

# Définir la dimension cible pour les images
target_dimension = (224, 224)  # Par exemple, 128x128 pixels

# Initialiser des listes pour stocker les images et les étiquettes
images = []
labels = []

# Fonction pour charger et redimensionner les images
def load_and_resize_images(folder_path, label):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        img = cv2.resize(img, target_dimension)  # Redimensionner l'image
        images.append(img)
        labels.append(label)

# Charger les images et étiqueter comme "bike"
load_and_resize_images(bike_folder, label="bike")

# Charger les images et étiqueter comme "car"
load_and_resize_images(car_folder, label="car")

# Convertir les listes d'images et d'étiquettes en tableaux NumPy
images = np.array(images)
labels = np.array(labels)
# %%
def populate_images_and_labels_lists(image_folder_path, target_dimension, label):
    images = []
    labels = []
    
    for filename in os.listdir(image_folder_path):
        image_path = os.path.join(image_folder_path, filename)
        img = cv2.imread(image_path)
        img = cv2.resize(img, target_dimension)  # Redimensionner l'image
        
        images.append(img)
        labels.append(label)
    
    return images, labels
# %%
import numpy as np

# Convertir les listes en tableaux NumPy
images_array = np.array(images)
labels_array = np.array(labels)

# Aplatir chaque image dans la liste d'images
flattened_images = [image.flatten() for image in images]

# Convertir la liste d'images aplaties en un tableau NumPy
images_array = np.array(flattened_images)


# %%
from sklearn.model_selection import train_test_split

# Spécifiez le pourcentage pour l'ensemble de test
test_size = 0.2

# Utilisez train_test_split pour séparer les données
X_train, X_test, y_train, y_test = train_test_split(images_array, labels_array, test_size=test_size, random_state=0)

# %%
from sklearn.tree import DecisionTreeClassifier

clf = DecisionTreeClassifier(random_state=0)
clf.fit(X_train, y_train) 
premiere_image_test = X_test[0].reshape(1, -1)  
prediction = clf.predict(premiere_image_test)

print("Label prédit pour la première image du set de test :", prediction)

# %%
from sklearn.svm import SVC


# Définir le modèle SVM
clf_svm = SVC(random_state=0)

# Entraîner le modèle SVM sur les données d'entraînement
clf_svm.fit(X_train, y_train)  # Assurez-vous d'avoir déjà défini X_train et y_train
# Assurez-vous que X_train et X_test sont au format numpy
X_train = np.array(X_train)
X_test = np.array(X_test)

# Redimensionnez les données si elles ne sont pas déjà bidimensionnelles
if len(X_train.shape) == 1:
    X_train = X_train.reshape(-1, 1)

if len(X_test.shape) == 1:
    X_test = X_test.reshape(-1, 1)

# Entraînez et prédisez avec les modèles
clf.fit(X_train, y_train)
clf_svm.fit(X_train, y_train)

premiere_image_test = X_test[0].reshape(1, -1)
prediction = clf.predict(premiere_image_test)
prediction_svm = clf_svm.predict(premiere_image_test)
# Prédire le label de la première image du set de test avec le modèle SVM
prediction_svm = clf_svm.predict(premiere_image_test)

print("Label prédit pour la première image du set de test avec le modèle SVM :", prediction_svm)

# %%
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, roc_curve, roc_auc_score
import matplotlib.pyplot as plt

# 1. Calculer l'accuracy des modèles
accuracy_model1 = accuracy_score(y_test, clf.predict(X_test))
accuracy_model2 = accuracy_score(y_test, clf_svm.predict(X_test))

print("Accuracy du modèle 1 :", accuracy_model1)
print("Accuracy du modèle 2 (SVM) :", accuracy_model2)

# 2. Calculer la matrice de confusion du modèle 1
confusion_matrix_model1 = confusion_matrix(y_test, clf.predict(X_test))
print("Matrice de confusion du modèle 1 :\n", confusion_matrix_model1)

# 3. Calculer la matrice de confusion du modèle 2 (SVM)
confusion_matrix_model2 = confusion_matrix(y_test, clf_svm.predict(X_test))
print("Matrice de confusion du modèle 2 (SVM) :\n", confusion_matrix_model2)

# Bonus: Calculer la précision et le rappel du modèle 1
precision_model1 = precision_score(y_test, clf.predict(X_test), pos_label='car')
recall_model1 = recall_score(y_test, clf.predict(X_test), pos_label='car')
print("Précision du modèle 1 :", precision_model1)
print("Rappel du modèle 1 :", recall_model1)

y_scores_model1 = clf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_scores_model1, pos_label='car')
roc_auc = roc_auc_score(y_test, y_scores_model1)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc='lower right')
plt.show()

# %%
from sklearn.tree import DecisionTreeClassifier

# Créer et entraîner un modèle d'arbre de décision
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# Obtenir la profondeur de l'arbre de décision
profondeur_arbre = clf.tree_.max_depth

print(f"La profondeur de l'arbre de décision est : {profondeur_arbre}")
# %%
max_depth_list = list(range(1, 13))
train_accuracy = []
test_accuracy = []

for max_depth in max_depth_list:
    # Créer et entraîner un modèle d'arbre de décision avec la profondeur maximale spécifiée
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)  # Assurez-vous d'utiliser vos données d'entraînement ici

    # Prédire sur les ensembles d'entraînement et de test
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)  # Assurez-vous d'utiliser vos données de test ici

    # Calculer l'accuracy et l'ajouter aux listes correspondantes
    train_accuracy.append(accuracy_score(y_train, y_train_pred))
    test_accuracy.append(accuracy_score(y_test, y_test_pred))

# Maintenant, train_accuracy[i] contient l'accuracy d'entraînement pour max_depth = max_depth_list[i]
# et test_accuracy[i] contient l'accuracy de test pour max_depth = max_depth_list[i]

# %%

# Créez le graphique
plt.plot(max_depth_list, train_accuracy, label='Accuracy d\'entraînement')
plt.plot(max_depth_list, test_accuracy, label='Accuracy de test')

# Ajoutez une légende pour les courbes
plt.legend()

# Ajoutez des étiquettes aux axes et un titre
plt.xlabel('max_depth')
plt.ylabel('Accuracy')
plt.title('Accuracy en fonction de max_depth')

# Affichez le graphique
plt.show()

# %%
import os
import cv2
import numpy as np

# Définissez le chemin du dossier de validation
validation_folder = r'C:\Users\hbass\OneDrive\Bureau\ComputerVision\val\bike'

# Définissez la dimension cible pour les images
target_dimension = (224, 224)  # Par exemple, 224x224 pixels

# Initialiser des listes pour stocker les images et les étiquettes de validation
val_images = []
val_labels = []

# Définissez une fonction pour charger et prétraiter les images
def load_and_preprocess_images(folder_path, label):
    for filename in os.listdir(folder_path):
        image_path = os.path.join(folder_path, filename)
        img = cv2.imread(image_path)
        img = cv2.resize(img, target_dimension)  # Redimensionnez l'image
        # Appliquez d'autres étapes de prétraitement ici si nécessaire
        val_images.append(img)
        val_labels.append(label)

# Chargez les images de validation et étiquetez-les (par exemple, "bike" et "car")
load_and_preprocess_images(os.path.join(validation_folder, 'bike'), label="bike")
load_and_preprocess_images(os.path.join(validation_folder, 'car'), label="car")

# Convertissez les listes d'images et d'étiquettes de validation en tableaux NumPy
val_images = np.array(val_images)
val_labels = np.array(val_labels)

from sklearn.metrics import accuracy_score

# Chargez le modèle avec la meilleure valeur de max_depth
best_max_depth =  0
clf = DecisionTreeClassifier(max_depth=best_max_depth)

# Utilisez le modèle pour prédire les étiquettes des données de validation
val_predictions = clf.predict(val_images.reshape(val_images.shape[0], -1))

# Calculez l'accuracy en comparant les étiquettes prédites avec les étiquettes réelles
accuracy = accuracy_score(val_labels, val_predictions)

print(f"Accuracy des données de validation : {accuracy:.2f}")
