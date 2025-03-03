import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Génération des données simulées de sondages et sauvegarde dans un fichier CSV
def generate_survey_data_and_save(filename="survey_data.csv"):
    # Simuler des coordonnées de sondages (x, y) et des teneurs en minerai
    np.random.seed(42)
    
    # 100 points de sondage dans une zone de 1000x1000 mètres
    x = np.random.uniform(0, 1000, 100)
    y = np.random.uniform(0, 1000, 100)
    
    # Teneur en minerai simulée, par exemple, entre 0 et 10%
    grade = np.random.uniform(0, 10, 100)
    
    # Organiser ces données dans un DataFrame
    data = pd.DataFrame({'x': x, 'y': y, 'grade': grade})
    
    # Sauvegarder les données dans un fichier CSV
    data.to_csv(filename, index=False)
    print(f"Les données ont été enregistrées dans le fichier {filename}")
    
    return data

# Interpolation des données à l'aide de GaussianProcessRegressor (méthode alternative à PyKrige)
def perform_gaussian_process_interpolation(data, grid_size=100):
    # Créer un maillage sur la zone d'intérêt (1000x1000m) avec un espacement de 10 mètres
    grid_x, grid_y = np.meshgrid(np.linspace(0, 1000, grid_size), np.linspace(0, 1000, grid_size))
    
    # Préparer les données d'entrée pour l'interpolation
    X = np.array(list(zip(data['x'], data['y'])))
    y = data['grade'].values
    
    # Définir le noyau (RBF) et l'algorithme de régression
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    
    # Ajuster le modèle
    gp.fit(X, y)
    
    # Prédire les teneurs sur le maillage
    Z, sigma = gp.predict(np.vstack([grid_x.ravel(), grid_y.ravel()]).T, return_std=True)
    
    # Remodeler les résultats pour obtenir une carte 2D
    Z = Z.reshape(grid_x.shape)
    
    return grid_x, grid_y, Z

# Visualisation de la carte de chaleur
def plot_heatmap(grid_x, grid_y, z):
    # Visualiser la carte de chaleur de la teneur en minerai
    plt.figure(figsize=(10, 8))
    plt.contourf(grid_x, grid_y, z, cmap='YlGnBu', levels=100)
    plt.colorbar(label='Teneur en minerai (%)')
    plt.title('Cartographie des teneurs en minerai')
    plt.xlabel('Coordonnée X (m)')
    plt.ylabel('Coordonnée Y (m)')
    plt.show()

# Sauvegarder les résultats d'interpolation dans un fichier CSV
def save_interpolated_data(grid_x, grid_y, z, filename="interpolated_data.csv"):
    # Créer un DataFrame avec les résultats
    result_df = pd.DataFrame({
        'x': grid_x.ravel(),
        'y': grid_y.ravel(),
        'grade': z.ravel()
    })
    
    # Sauvegarder les résultats dans un fichier CSV
    result_df.to_csv(filename, index=False)
    print(f"Les résultats d'interpolation ont été enregistrés dans {filename}")
    
    return result_df

# Fonction principale pour orchestrer les étapes
def main():
    # Étape 1 : Générer les données de sondage et les enregistrer dans un fichier CSV
    data = generate_survey_data_and_save()
    
    # Étape 2 : Appliquer l’interpolation géostatistique avec GaussianProcessRegressor
    grid_x, grid_y, z = perform_gaussian_process_interpolation(data)
    
    # Étape 3 : Visualiser la cartographie des teneurs en minerai
    plot_heatmap(grid_x, grid_y, z)
    
    # Étape 4 : Sauvegarder les résultats dans un fichier CSV
    save_interpolated_data(grid_x, grid_y, z)

# Exécution du projet
if __name__ == '__main__':
    main()
