# Proyecto de Clustering con K-Means

Este proyecto utiliza el algoritmo de clustering K-Means para clasificar diferentes conjuntos de datos. Los usuarios pueden ingresar el número de clústeres y seleccionar los conjuntos de datos que desean generar y clasificar.

## Requisitos

- Python 3.7 o superior
- pip

## Instalación

1. Clona este repositorio:

    ```bash
    git clone https://github.com/zaphod9801/kmeans_sistemas_autoadaptables
    cd kmeans_sistemas_autoadaptables
    ```

2. Crea un entorno virtual y actívalo:

    - En Windows:

        ```bash
        python -m venv venv
        venv\Scripts\activate
        ```

    - En macOS y Linux:

        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```

3. Instala las dependencias:

    ```bash
    pip install -r requirements.txt
    ```

## Uso

1. Asegúrate de que el entorno virtual esté activado.

2. Ejecuta el script principal `kmeans.py`:

    ```bash
    python kmeans.py
    ```

3. Sigue las instrucciones en la consola para ingresar el número de clústeres y seleccionar los conjuntos de datos que deseas clasificar.

## Conjuntos de Datos Disponibles

- `noisy_circles`
- `noisy_moons`
- `varied`
- `aniso`
- `blobs`
- `no_structure`

## Notas

- Asegúrate de tener todos los paquetes necesarios instalados.
- Si encuentras algún problema, revisa que estás usando la versión correcta de Python y que el entorno virtual está activado.

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.
