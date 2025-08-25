# graph_extractor.py

import cv2
import numpy as np
import csv

def extract_and_save_curve(image_path, output_csv_path, lower_color, upper_color, axis_calibration):
    """
    Extrait les points d'une courbe d'une couleur spécifique d'une image,
    les convertit en coordonnées de données et les sauvegarde dans un fichier CSV.

    Args:
        image_path (str): Chemin vers l'image du graphique.
        output_csv_path (str): Chemin où sauvegarder le fichier CSV.
        lower_color (np.array): Limite inférieure de la couleur à extraire (en HSV).
        upper_color (np.array): Limite supérieure de la couleur à extraire (en HSV).
        axis_calibration (dict): Dictionnaire contenant les points de calibration.
    """
    # 1. Charger l'image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Erreur: Impossible de charger l'image depuis {image_path}")
        return

    # 2. Convertir en HSV pour une meilleure détection de couleur
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # 3. Créer un masque pour la couleur de la courbe
    mask = cv2.inRange(hsv, lower_color, upper_color)

    # 4. Trouver les coordonnées des pixels de la courbe
    # findNonZero retourne les coordonnées (colonne, ligne) des pixels non-nuls (blancs)
    pixel_coords = cv2.findNonZero(mask)
    if pixel_coords is None:
        print(f"Aucune courbe détectée pour la plage de couleur définie dans {output_csv_path}.")
        return

    # Les coordonnées sont retournées en (x, y) qui est (colonne, ligne).
    # Pour le traitement, y est souvent inversé (0 en haut).
    pixel_points = pixel_coords.reshape(-1, 2)

    # 5. Calibration et Conversion
    px_min, py_min = axis_calibration['pixel_origin']
    px_max, py_max = axis_calibration['pixel_top_right']
    
    data_x_min, data_y_min = axis_calibration['data_origin']
    data_x_max, data_y_max = axis_calibration['data_top_right']

    # Calcul des ratios de conversion
    x_scale = (data_x_max - data_x_min) / (px_max - px_min)
    y_scale = (data_y_max - data_y_min) / (py_max - py_min)

    # Conversion des points
    # Note: pour l'axe Y des pixels, l'origine est en haut, donc on l'inverse.
    data_points = []
    for x, y in pixel_points:
        data_x = data_x_min + (x - px_min) * x_scale
        data_y = data_y_min + (py_max - y) * y_scale # Inversion de l'axe Y du pixel
        data_points.append((data_x, data_y))
        
    # Trier les points par la coordonnée X pour avoir une belle courbe
    data_points.sort()

    # 6. Sauvegarder en CSV
    with open(output_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['x', 'y'])
        writer.writerows(data_points)
    
    print(f"Extraction réussie ! {len(data_points)} points sauvegardés dans {output_csv_path}")


# --- CONFIGURATION ---

# TODO: À MODIFIER PAR L'UTILISATEUR
# 1. Chemin de l'image
IMAGE_FILE = 'Biblio.png'

# 2. Calibration des axes : C'EST L'ÉTAPE LA PLUS IMPORTANTE !
#    Vous devez trouver les coordonnées en PIXELS de l'origine (coin bas-gauche)
#    et du point max (haut-droit) de votre ZONE DE TRACÉ.
#    Utilisez un éditeur d'image (comme Paint, GIMP) pour trouver ces pixels.
#    (0,0) est le coin SUPÉRIEUR GAUCHE de l'image.
AXIS_CALIBRATION = {
    # Coordonnées en PIXELS
    'pixel_origin':    (50, 450),  # (x, y) du coin bas-gauche du graphique
    'pixel_top_right': (800, 50), # (x, y) du coin haut-droit du graphique

    # Valeurs de DONNÉES correspondantes
    'data_origin':     (0, 0),     # (valeur_x, valeur_y) à l'origine
    'data_top_right':  (100, 250)  # (valeur_x, valeur_y) au point max
}

# 3. Définition des couleurs des courbes à extraire (en HSV)
#    Astuce: Pour trouver les bonnes valeurs, vous pouvez ouvrir l'image
#    dans un logiciel comme GIMP, utiliser le sélecteur de couleur et noter les valeurs HSV.
#    HSV = (Teinte, Saturation, Valeur/Luminosité)
#    Teinte: 0-179, Saturation: 0-255, Valeur: 0-255 en OpenCV
COLORS_TO_EXTRACT = {
    "serie_A": {
        "lower": np.array([100, 150, 50]),  # Bleu - plage inférieure
        "upper": np.array([130, 255, 255])  # Bleu - plage supérieure
    },
    "serie_B": {
        "lower": np.array([0, 150, 50]),    # Rouge/Orange - plage inférieure
        "upper": np.array([10, 255, 255])   # Rouge/Orange - plage supérieure
    }
    # Ajoutez d'autres couleurs ici
}

# --- EXÉCUTION ---
if __name__ == "__main__":
    for name, colors in COLORS_TO_EXTRACT.items():
        output_file = f"{name}_extracted.csv"
        extract_and_save_curve(
            image_path=IMAGE_FILE,
            output_csv_path=output_file,
            lower_color=colors['lower'],
            upper_color=colors['upper'],
            axis_calibration=AXIS_CALIBRATION
        )
