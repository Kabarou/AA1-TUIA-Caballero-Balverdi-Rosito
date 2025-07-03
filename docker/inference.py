import os
import joblib
import pandas as pd
import numpy as np
import logging
from sys import stdout

# Importar las clases custom para que joblib encuentre las definiciones al deserializar
from create_pipeline import (
    MonthExtractor,
    DropNullCoords,
    AddCoordinates,
    AssignRegion,
    WindDirTransformer,
    CategoricalRFImputer,
)

# Configuración del logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s %(levelname)s %(filename)s: %(message)s')
ch = logging.StreamHandler(stdout)
ch.setFormatter(fmt)
logger.addHandler(ch)

# Ruta al pipeline serializado
MODEL_PKL = os.getenv("MODEL_PKL", "docker/files/pipeline.pkl")
logger.info(f"Cargando pipeline desde {MODEL_PKL}")
pipeline = joblib.load(MODEL_PKL)

# Leer datos de entrada
INPUT_CSV = os.getenv("INPUT_CSV", "docker/files/input.csv") # Reemplazar con la ruta absoluta
df_input = pd.read_csv(INPUT_CSV)
logger.info(f"Cargado input: {df_input.shape[0]} filas")

# Predecir
proba = pipeline.predict_proba(df_input)[:, 1]
preds = pipeline.predict(df_input)
df_out = pd.DataFrame({
    "prediction": preds,
    "probability": proba
})

# Guardar salida
OUTPUT_CSV = os.getenv("OUTPUT_CSV", "docker/files/output.csv")
df_out.to_csv(OUTPUT_CSV, index=False)
logger.info(f"Predicciones guardadas en {OUTPUT_CSV}")

