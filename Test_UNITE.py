pip install ucimlrepo
# Importer les bibliothèques nécessaires
import pandas as pd
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo

# Charger le jeu de données "Statlog German Credit"
data = fetch_ucirepo(id=144)

# Extraire les données et les métadonnées
df = data.data  # Les données sous forme de DataFrame Pandas
metadata = data.metadata  # Métadonnées
variables_info = data.variables[['name', 'description']]  # Description des variables

# Aperçu des données
print("Aperçu des données :")
print(df.head())

# Description des variables
print("\nDescription des variables :")
print(variables_info)

# Statistiques descriptives
print("\nStatistiques descriptives :")
print(df.describe())

# Visualisation - Histogrammes pour les distributions
df.hist(figsize=(15, 10), bins=20)
plt.tight_layout()
plt.show()
