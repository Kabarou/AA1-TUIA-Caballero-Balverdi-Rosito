# Predicción de lluvia en Australia con Machine Learning (AA1–TUIA)   

Este repositorio contiene un pipeline en Python para la clasificación de lluvia y una imagen Docker para inferencia.


1. Clona el repositorio:
   ```bash
   git clone https://github.com/Kabarou/AA1-TUIA-Caballero-Balverdi-Rosito.git
   ```

2. Genera el pipeline serializado (`pipeline.pkl`):
   ```bash
   python docker/create_pipeline.py
   ```
   > **Nota:** `pipeline.pkl` excede los 100 MB y por eso no se incluye en GitHub.

3. Desde la raíz del repositorio, construye la imagen:
    ```bash
    docker build   -t aa1-clasificacion:latest   -f docker/dockerfile .
    ```

4. Copia tu fichero de entrada (`input.csv`) en `docker/files/` (o usa el que ya está: `docker/files/input.csv`).

5. Ejecuta el contenedor, montando la carpeta `files` para las entradas y salidas:
    ```bash
    docker run --rm \
    -m 4g \
    -v "$(pwd)/docker/files:/app/docker/files" \
    -e MODEL_PKL=/app/docker/files/pipeline.pkl \
    -e INPUT_CSV=/app/docker/files/input.csv \
    -e OUTPUT_CSV=/app/docker/files/output.csv \
    aa1-clasificacion:latest
    ```
    - Esto leerá `docker/files/input.csv` dentro del contenedor.  
    - Las predicciones se volcarán en `docker/files/output.csv` en tu máquina.

6. Consulta los resultados en `docker/files/output.csv`.  