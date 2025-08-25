# ERRORGRAPH — The Matrix of Graphs  
*Auteur : Idir ARSLANE — © 2025*

```
████████████████████████████████████████████████████████████████████████
█  01000101 01010010 01010010 01001111 01010010 01000111 01010010 01000001 █
█                         E R R O R G R A P H                     █
█                     The Matrix of Graphs • © 2025                       █
█                         Auteur : Idir ARSLANE                           █
████████████████████████████████████████████████████████████████████████
```

---

## 🕶️ Présentation
**ErrorGraph** est une suite d’outils Python permettant :  
- d’extraire des courbes à partir d’images (PDF, figures, scans),  
- de comparer ces données avec des modèles numériques ou expérimentaux,  
- de calculer automatiquement les erreurs (**RMSE, MAE, MAPE, R²**),  
- d’exporter en **CSV, PNG, XLSX, MAT**,  
- d’afficher les résultats dans une **UI Matrix** animée avec pluie de symboles verts.

---

## 📂 Contenu du projet

- **`Graph_ui.py`** : interface graphique Matrix (Tkinter + Matplotlib), pipette, calibration, extraction robuste, comparaison et exports.  
- **`robust_extract.py`** : extraction robuste par couleur LAB + squelette, pipeline CLI avec exports debug.  
- **`graph_extractor.py`** : version simple basée sur HSV + calibration manuelle.  
- **`graph.py`** : comparaison auto de deux graphes, métriques et exports.  
- **`graph_vision.py`** : OCR (ticks, axes, échelles) + segmentation multi-courbes (K-means).  
- **`view_results.py`** : affiche les résultats produits (`overlay.png`, `error_percent.png`, `compare_resampled.csv`).  
- **`.gitignore`** : exclusions (cache Python, builds, settings).  
- **`requirements.txt`** : dépendances Python.  

---

## ⚡ Installation

```bash
git clone https://github.com/<user>/ErrorGraph.git
cd ErrorGraph

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
```

Dépendances principales :  
`numpy, opencv-python, scikit-image, matplotlib, Pillow, pytesseract, pandas, openpyxl, scipy`

---

## 🖥️ Utilisation rapide

### Mode UI (Matrix)
```bash
python Graph_ui.py
```
- Charger 2 graphes (image ou tableau CSV/XLSX/MAT)  
- Calibration par clic (origine + coin haut-droit)  
- Extraction → Superposition → Calcul erreurs  
- Export PNG + CSV  

### Mode CLI robuste
```bash
python robust_extract.py --img courbe.png --hex "#1F497D" --outdir out --debug
```

### Comparaison auto
```bash
python graph.py --img-a Biblio.png --img-b Modele.png --color-a blue --color-b blue --outdir out
```

---

## 📊 Résultats

- **overlay.png** → superposition Biblio vs Modèle  
- **error_percent.png** → courbe Erreur relative (%)  
- **series_A.csv / series_B.csv** → données extraites  
- **compare_resampled.csv** → tableau comparatif (A, B, erreur absolue, erreur %)  

---

## 🧩 Avancées
- OCR automatique des graduations pour conversion en unités physiques (`graph_vision.py`).  
- Multi-courbes (segmentation K-means).  
- Sauvegarde des paramètres (`MySettings/last_settings.json`).  
- Thème **Matrix** avec bandeau pluie animée en arrière-plan.  

---

## ☣️ Auteur
Projet développé par **Idir ARSLANE**

```
>> Initializing ErrorGraph...
>> Ready for extraction. Follow the white rabbit.
```
