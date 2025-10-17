# Rapport Synthétique – Prédiction du Temps de Livraison

## 1. Contexte
L’objectif du projet est de prédire le **temps total de livraison** d’une commande, depuis la préparation jusqu’à la réception.  
Les données disponibles comprennent : distance, type de véhicule, trafic, météo, expérience du livreur, temps de préparation, etc.  

L’enjeu principal est d’**optimiser les livraisons** et d’améliorer la satisfaction client.

---

## 2. Données
- **Source** : fichier CSV `Dataa.csv`
- **Dimensions** : 1000 lignes × 9 colonnes
- **Variables clés** :
  - `Distance_km`
  - `Weather`
  - `Traffic_Level`
  - `Time_of_Day`
  - `Vehicle_Type`
  - `Preparation_Time_min`
  - `Courier_Experience_yrs`
  - `Delivery_Time_min` (variable cible)
- **Problèmes détectés** :
  - Valeurs manquantes (`Weather`, `Traffic_Level`, `Time_of_Day`, `Courier_Experience_yrs`)
  - Outliers sur certaines variables numériques
- **Prétraitement** :
  - Remplacement des valeurs manquantes :
    - Moyenne pour les variables numériques
    - Mode pour les variables catégorielles
  - Encodage des variables catégorielles (`LabelEncoder`)
  - Suppression des outliers via la méthode IQR

---

## 3. Exploration des données (EDA)
- Visualisation et analyses :
  - Heatmap pour les corrélations
  - Boxplots pour les relations entre variables catégorielles et temps de livraison
  - Histogrammes et scatterplots pour la distribution de la variable cible et les relations avec la distance
- **Sélection des features** :
  - `SelectKBest` + `f_regression`
  - 4 features les plus pertinentes retenues pour la modélisation finale

---

## 4. Modélisation
- **Pipeline scikit-learn** :
  - Prétraitement (StandardScaler + OneHotEncoder)
  - Sélection de features (`SelectKBest`)
  - Modèle prédictif
- **Modèles testés** :
  - Random Forest Regressor
  - Support Vector Regressor (SVR)
- **Optimisation** :
  - GridSearchCV pour hyperparamètres
  - Standardisation des données
- **Évaluation** :
  - MAE (Mean Absolute Error)
  - R² (coefficient de détermination)

---

## 5. Résultats
- **Meilleur modèle** : Random Forest / SVR selon MAE
- **Performance finale** :
  - MAE ≈ `XX` minutes
  - R² ≈ `XX`
- Les performances sont jugées satisfaisantes pour des prédictions opérationnelles.

---

## 6. Conclusion
- Le modèle permet de prédire le temps de livraison avec une précision correcte.
- Applications possibles :
  - Anticiper les retards de livraison
  - Optimiser la planification des courriers et des véhicules
- Perspectives d’amélioration :
  - Ajouter plus de données historiques
  - Tester d’autres modèles (XGBoost, LightGBM)
  - Intégrer la météo et le trafic en temps réel
