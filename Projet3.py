import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C

# Génération des données simulées de forages et sauvegarde dans un fichier CSV
def generate_drilling_data_and_save(filename="drilling_data.csv"):
    np.random.seed(42)
    
    # Simuler des coordonnées de sondages (x, y, z) et des teneurs en minerai
    x = np.random.uniform(0, 1000, 100)  # Coordonnées x dans une zone de 1000x1000 m
    y = np.random.uniform(0, 1000, 100)  # Coordonnées y
    z = np.random.uniform(0, 500, 100)  # Profondeur des sondages (0 à 500 m)
    
    # Teneur en minerai simulée, par exemple entre 0 et 10 %
    grade = np.random.uniform(0, 10, 100)
    
    # Organiser ces données dans un DataFrame
    data = pd.DataFrame({'x': x, 'y': y, 'z': z, 'grade': grade})
    
    # Sauvegarder les données dans un fichier CSV
    data.to_csv(filename, index=False)
    print(f"Les données ont été enregistrées dans le fichier {filename}")
    
    return data

# Modélisation 3D du corps minéralisé avec interpolation
def perform_3d_interpolation(data, grid_size=50):
    # Créer un maillage 3D pour les coordonnées x, y, z
    grid_x, grid_y, grid_z = np.meshgrid(np.linspace(0, 1000, grid_size),
                                          np.linspace(0, 1000, grid_size),
                                          np.linspace(0, 500, grid_size))
    
    # Préparer les données d'entrée pour l'interpolation (x, y, z)
    X = np.array(list(zip(data['x'], data['y'], data['z'])))
    y = data['grade'].values
    
    # Définir le noyau pour GaussianProcessRegressor
    kernel = C(1.0, (1e-4, 1e1)) * RBF(1.0, (1e-4, 1e1))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2)
    
    # Ajuster le modèle
    gp.fit(X, y)
    
    # Prédire les teneurs sur le maillage 3D
    grid_points = np.vstack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()]).T
    Z, sigma = gp.predict(grid_points, return_std=True)
    
    # Remodeler les résultats pour obtenir une matrice 3D
    Z = Z.reshape(grid_x.shape)
    
    return grid_x, grid_y, grid_z, Z

# Visualisation 3D du modèle du corps minéralisé
def plot_3d_model(grid_x, grid_y, grid_z, z):
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # Tracer le modèle 3D
    ax.scatter(grid_x, grid_y, grid_z, c=z, cmap='viridis', marker='o', s=5)
    
    ax.set_title('Modélisation 3D du corps minéralisé')
    ax.set_xlabel('Coordonnée X (m)')
    ax.set_ylabel('Coordonnée Y (m)')
    ax.set_zlabel('Profondeur (m)')
    plt.show()

# Sauvegarder les résultats d'interpolation 3D dans un fichier CSV
def save_3d_interpolated_data(grid_x, grid_y, grid_z, z, filename="interpolated_3d_data.csv"):
    # Créer un DataFrame avec les résultats
    result_df = pd.DataFrame({
        'x': grid_x.ravel(),
        'y': grid_y.ravel(),
        'z': grid_z.ravel(),
        'grade': z.ravel()
    })
    
    # Sauvegarder les résultats dans un fichier CSV
    result_df.to_csv(filename, index=False)
    print(f"Les résultats d'interpolation 3D ont été enregistrés dans {filename}")
    
    return result_df

# Fonction principale pour orchestrer les étapes
def main():
    # Étape 1 : Générer les données de forages et les enregistrer dans un fichier CSV
    data = generate_drilling_data_and_save()
    
    # Étape 2 : Appliquer l’interpolation 3D avec GaussianProcessRegressor
    grid_x, grid_y, grid_z, z = perform_3d_interpolation(data)
    
    # Étape 3 : Visualiser la modélisation 3D du corps minéralisé
    plot_3d_model(grid_x, grid_y, grid_z, z)
    
    # Étape 4 : Sauvegarder les résultats d'interpolation 3D dans un fichier CSV
    save_3d_interpolated_data(grid_x, grid_y, grid_z, z)

# Exécution du projet
if __name__ == '__main__':
    main()
