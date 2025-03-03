import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform, cdist

# ==============================
# 1. Chargement et Visualisation des Données
# ==============================

def charger_donnees(fichier):
    """ Charge les données des sondages à partir d'un fichier CSV. """
    df = pd.read_csv(fichier)
    print(df.head())  # Affichage des premières lignes
    print(df.describe())  # Statistiques descriptives
    return df

def afficher_distribution_teneurs(df):
    """ Affiche l'histogramme des teneurs en minerai. """
    plt.figure(figsize=(8,5))
    plt.hist(df['Teneur'], bins=30, color='blue', edgecolor='black', alpha=0.7)
    plt.xlabel("Teneur (%)")
    plt.ylabel("Fréquence")
    plt.title("Distribution des teneurs en minerai")
    plt.show()

# ==============================
# 2. Calcul et Tracé du Variogramme
# ==============================

def calcul_variogramme(df, max_dist=200, nb_lags=15):
    """ Calcule un variogramme expérimental basé sur les distances entre forages. """
    dist_matrix = squareform(pdist(df[['X', 'Y']].values))
    valeur_matrix = np.subtract.outer(df['Teneur'].values, df['Teneur'].values)**2

    bins = np.linspace(0, max_dist, nb_lags)
    bin_means = np.zeros(len(bins)-1)

    for i in range(len(bins)-1):
        mask = (dist_matrix > bins[i]) & (dist_matrix <= bins[i+1])
        bin_means[i] = np.mean(valeur_matrix[mask])

    return bins[:-1], bin_means

def afficher_variogramme(lags, variog):
    """ Affiche le variogramme expérimental. """
    plt.figure(figsize=(8,5))
    plt.scatter(lags, variog, label="Variogramme expérimental", color="red")
    plt.xlabel("Distance (m)")
    plt.ylabel("Semi-variance")
    plt.title("Variogramme expérimental des teneurs")
    plt.legend()
    plt.show()

# ==============================
# 3. Interpolation IDW
# ==============================

def idw_interpolation(x, y, df, power=2):
    """ Interpolation des teneurs par Inverse Distance Weighting (IDW). """
    points = df[['X', 'Y']].values
    valeurs = df['Teneur'].values
    distances = cdist(np.array([[x, y]]), points)[0]
    distances[distances < 1e-6] = 1e-6  # Éviter division par zéro

    poids = 1 / (distances ** power)
    teneur_interpolee = np.sum(valeurs * poids) / np.sum(poids)
    
    return teneur_interpolee

def afficher_interpolation(df):
    """ Crée une grille d'interpolation et affiche la carte interpolée des teneurs. """
    x_range = np.linspace(df["X"].min(), df["X"].max(), 50)
    y_range = np.linspace(df["Y"].min(), df["Y"].max(), 50)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    Z_interp = np.vectorize(idw_interpolation)(X_grid, Y_grid, df)

    plt.figure(figsize=(8,6))
    plt.contourf(X_grid, Y_grid, Z_interp, cmap='inferno', levels=20)
    plt.colorbar(label="Teneur (%)")
    plt.scatter(df["X"], df["Y"], c=df["Teneur"], edgecolors='k', cmap='inferno')
    plt.xlabel("X (m)")
    plt.ylabel("Y (m)")
    plt.title("Interpolation des teneurs par IDW")
    plt.show()

# ==============================
# 4. Estimation des Réserves Minières
# ==============================

def estimer_reserves(df, bloc_size=50, densite=2.7):
    """ Estimation des réserves minières en utilisant un modèle de blocs et l'IDW. """
    x_range = np.arange(df["X"].min(), df["X"].max(), bloc_size)
    y_range = np.arange(df["Y"].min(), df["Y"].max(), bloc_size)
    z_range = np.arange(df["Z"].min(), df["Z"].max(), bloc_size)

    blocs = np.array([[x, y, z] for x in x_range for y in y_range for z in z_range])
    bloc_teneurs = np.array([idw_interpolation(x, y, df) for x, y, z in blocs])

    volume_bloc = bloc_size**3  # Volume en m³
    reserve_totale = np.sum(bloc_teneurs * densite * volume_bloc / 100)  # en tonnes

    print(f"Réserves estimées: {reserve_totale:.2f} tonnes de minerai")
    return reserve_totale

# ==============================
# 5. Exécution du Programme Principal
# ==============================

if __name__ == "__main__":
    fichier_donnees = "donnees/donnees_sondage.csv"

    # Étape 1 : Chargement et analyse des données
    df = charger_donnees(fichier_donnees)
    afficher_distribution_teneurs(df)

    # Étape 2 : Calcul et tracé du variogramme
    lags, variog = calcul_variogramme(df)
    afficher_variogramme(lags, variog)

    # Étape 3 : Interpolation IDW
    afficher_interpolation(df)

    # Étape 4 : Estimation des réserves
    estimer_reserves(df)
