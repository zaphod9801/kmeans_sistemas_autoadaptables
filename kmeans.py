import time
import warnings
from itertools import cycle, islice

import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster, datasets
from sklearn.preprocessing import StandardScaler

# Función para obtener la entrada del usuario
def obtener_parametros():
    try:
        k = int(input("Ingrese el número de clústeres (k): "))
        if k <= 0:
            raise ValueError("El número de clústeres debe ser mayor que 0.")
    except ValueError as e:
        print(e)
        return obtener_parametros()
    
    opciones_figuras = {
        "1": "noisy_circles",
        "2": "noisy_moons",
        "3": "varied",
        "4": "aniso",
        "5": "blobs",
        "6": "no_structure",
    }
    
    print("\nSeleccione las figuras a generar (puede seleccionar múltiples separadas por comas):")
    print("1: noisy_circles")
    print("2: noisy_moons")
    print("3: varied")
    print("4: aniso")
    print("5: blobs")
    print("6: no_structure")
    
    seleccion = input("Ingrese los números correspondientes: ").split(",")
    
    figuras_seleccionadas = []
    for opcion in seleccion:
        opcion = opcion.strip()
        if opcion in opciones_figuras:
            figuras_seleccionadas.append(opciones_figuras[opcion])
        else:
            print(f"Opción inválida: {opcion}")
    
    if not figuras_seleccionadas:
        print("No se seleccionaron figuras válidas. Intente de nuevo.")
        return obtener_parametros()
    
    return k, figuras_seleccionadas

# Parámetros iniciales
n_samples = 500
seed = 30

# Conjuntos de datos disponibles
disponibles = {
    "noisy_circles": datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=seed),
    "noisy_moons": datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=seed),
    "blobs": datasets.make_blobs(n_samples=n_samples, random_state=seed),
    "no_structure": (np.random.RandomState(seed).rand(n_samples, 2), None),
    "varied": datasets.make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170),
    "aniso": (np.dot(datasets.make_blobs(n_samples=n_samples, random_state=170)[0], [[0.6, -0.6], [-0.4, 0.8]]), datasets.make_blobs(n_samples=n_samples, random_state=170)[1]),
}

# Obtener parámetros del usuario
k, figuras_seleccionadas = obtener_parametros()

# Configurar parámetros de clúster
plt.figure(figsize=(9, 13))
plt.subplots_adjust(left=0.02, right=0.98, bottom=0.001, top=0.95, wspace=0.05, hspace=0.01)

plot_num = 1

for nombre_figura in figuras_seleccionadas:
    X, y = disponibles[nombre_figura]

    # Normalizar el conjunto de datos para facilitar la selección de parámetros
    X = StandardScaler().fit_transform(X)

    # Crear objeto de clúster KMeans
    kmeans = cluster.MiniBatchKMeans(n_clusters=k, random_state=42)

    t0 = time.time()

    # Capturar advertencias relacionadas con kneighbors_graph
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the number of connected components of the connectivity matrix is [0-9]{1,2} > 1. Completing it to avoid stopping the tree early.",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message="Graph is not fully connected, spectral embedding may not work as expected.",
            category=UserWarning,
        )
        kmeans.fit(X)

    t1 = time.time()
    if hasattr(kmeans, "labels_"):
        y_pred = kmeans.labels_.astype(int)
    else:
        y_pred = kmeans.predict(X)

    plt.subplot(len(figuras_seleccionadas), 1, plot_num)
    if plot_num == 1:
        plt.title("KMeans", size=18)

    colors = np.array(
        list(
            islice(
                cycle(
                    [
                        "#377eb8",
                        "#ff7f00",
                        "#4daf4a",
                        "#f781bf",
                        "#a65628",
                        "#984ea3",
                        "#999999",
                        "#e41a1c",
                        "#dede00",
                    ]
                ),
                int(max(y_pred) + 1),
            )
        )
    )
    # Añadir color negro para outliers (si los hay)
    colors = np.append(colors, ["#000000"])
    plt.scatter(X[:, 0], X[:, 1], s=10, color=colors[y_pred])

    plt.xlim(-2.5, 2.5)
    plt.ylim(-2.5, 2.5)
    plt.xticks(())
    plt.yticks(())
    plt.text(
        0.99,
        0.01,
        ("%.2fs" % (t1 - t0)).lstrip("0"),
        transform=plt.gca().transAxes,
        size=15,
        horizontalalignment="right",
    )
    plot_num += 1

plt.show()
