import os
import joblib
import pandas as pd
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    StandardScaler, OneHotEncoder, PolynomialFeatures
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# 1) Coordenadas de cada ciudad
coordenadas_ciudades = {
    'Albury': (-36.08, 146.92),
    'Sydney': (-33.87, 151.21),
    'Melbourne': (-37.81, 144.96),
    'Brisbane': (-27.47, 153.02),
    'Perth': (-31.95, 115.86),
    'Adelaide': (-34.93, 138.60),
    'Darwin': (-12.46, 130.84),
    'Hobart': (-42.88, 147.33),
    'Canberra': (-35.28, 149.13),
    'MountGinini': (-35.53, 148.77),
    'GoldCoast': (-28.02, 153.40),
    'Wollongong': (-34.43, 150.89),
    'MountGambier': (-37.83, 140.78),
    'Launceston': (-41.44, 147.14),
    'AliceSprings': (-23.70, 133.88),
    'Albany': (-35.02, 117.88),
    'Townsville': (-19.26, 146.82),
    'Bendigo': (-36.76, 144.28),
    'Cairns': (-16.92, 145.77),
    'Ballarat': (-37.56, 143.85),
    'Penrith': (-33.75, 150.69),
    'Newcastle': (-32.93, 151.78),
    'Tuggeranong': (-35.42, 149.07),
    'PerthAirport': (-31.94, 115.97),
    'SalmonGums': (-32.98, 121.63),
    'Nhil': (-36.33, 141.65),
    'Katherine': (-14.47, 132.27),
    'Uluru': (-25.34, 131.03),
    'BadgerysCreek': (-33.92, 150.78),
    'Cobar': (-31.49, 145.83),
    'CoffsHarbour': (-30.30, 153.11),
    'Moree': (-29.46, 149.84),
    'NorahHead': (-33.28, 151.58),
    'NorfolkIsland': (-29.04, 167.95),
    'Richmond': (-33.60, 150.75),
    'SydneyAirport': (-33.94, 151.18),
    'WaggaWagga': (-35.12, 147.37),
    'Williamtown': (-32.79, 151.84),
    'Sale': (-38.11, 147.07),
    'MelbourneAirport': (-37.67, 144.84),
    'Mildura': (-34.19, 142.16),
    'Portland': (-38.34, 141.61),
    'Watsonia': (-37.71, 145.08),
    'Dartmoor': (-37.92, 141.27),
    'Nuriootpa': (-34.47, 138.99),
    'Woomera': (-31.15, 136.80),
    'Witchcliffe': (-34.00, 115.10),
    'PearceRAAF': (-31.67, 116.02),
    'Walpole': (-34.98, 116.73),
}
# 2) Transformers
class MonthExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        df['Date'] = pd.to_datetime(df['Date'])
        df['Month'] = df['Date'].dt.month
        return df.drop(columns=['Date'])

class AddCoordinates(BaseEstimator, TransformerMixin):
    def __init__(self, coord_dict): self.coord = coord_dict
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        df['Lat'] = df['Location'].map(lambda l: self.coord.get(l,(np.nan,np.nan))[0])
        df['Lon'] = df['Location'].map(lambda l: self.coord.get(l,(np.nan,np.nan))[1])
        return df

class DropNullCoords(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None): return self
    def transform(self, X):
        return X.dropna(subset=['Lat','Lon']).copy()

class AssignRegion(BaseEstimator, TransformerMixin):
    def __init__(self, n_clusters=4, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
    def fit(self, X, y=None):
        coords = X[['Lat','Lon']].values
        self.km_ = KMeans(n_clusters=self.n_clusters,
                          random_state=self.random_state).fit(coords)
        return self
    def transform(self, X):
        df = X.copy()
        df['Region'] = self.km_.predict(df[['Lat','Lon']].values)
        return df.drop(columns=['Lat','Lon'])

class WindDirTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.map_deg = {
            'N':0,'NNE':22.5,'NE':45,'ENE':67.5,'E':90,'ESE':112.5,'SE':135,
            'SSE':157.5,'S':180,'SSW':202.5,'SW':225,'WSW':247.5,'W':270,
            'WNW':292.5,'NW':315,'NNW':337.5
        }
    def fit(self, X, y=None): return self
    def transform(self, X):
        df = X.copy()
        for col in ['WindGustDir','WindDir9am','WindDir3pm']:
            deg = df[col].map(self.map_deg).fillna(0)
            df[f'{col}_sin'] = np.sin(np.deg2rad(deg))
            df[f'{col}_cos'] = np.cos(np.deg2rad(deg))
        return df.drop(columns=['WindGustDir','WindDir9am','WindDir3pm'])

class CategoricalRFImputer(BaseEstimator, TransformerMixin):
    def __init__(self, cat_cols, num_cols):
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.models = {}

    def fit(self, X, y=None):
        df = X.copy()
        for col in self.cat_cols:
            mask = df[col].notnull()
            if mask.any():
                X_train = df.loc[mask, self.num_cols]
                y_train = df.loc[mask, col]
                self.models[col] = RandomForestClassifier(n_estimators=100, random_state=42) \
                                    .fit(X_train, y_train)
        return self

    def transform(self, X):
        df = X.copy()
        for col, model in self.models.items():
            mask = df[col].isnull()
            if mask.any():
                X_pred = df.loc[mask, self.num_cols]
                df.loc[mask, col] = model.predict(X_pred)
        return df

# 3) Columnas
num_cols = ['MinTemp','MaxTemp','Rainfall','Evaporation','Sunshine',
            'WindGustSpeed','WindSpeed9am','WindSpeed3pm',
            'Humidity9am','Humidity3pm','Pressure9am','Pressure3pm',
            'Temp9am','Temp3pm']
cat_cols = ['RainToday','Region','Cloud9am','Cloud3pm','Month']

# 4) Preprocesamiento
numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler',   StandardScaler()),
    ('poly',     PolynomialFeatures(degree=3,
                                   interaction_only=True,
                                   include_bias=False))
])
categorical_pipeline = Pipeline([
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
preprocessor = ColumnTransformer([
    ('num', numeric_pipeline, num_cols),
    ('cat', categorical_pipeline, cat_cols),
], remainder='drop')

# 5) Construcción del pipeline final
pipeline = Pipeline([
    ('month',     MonthExtractor()),
    ('coords',    AddCoordinates(coordenadas_ciudades)),
    ('drop_null', DropNullCoords()),
    ('region',    AssignRegion()),
    ('wind_dir',  WindDirTransformer()),
    ('cat_imp',   CategoricalRFImputer(cat_cols, num_cols)),  # <-- imputación RF aquí
    ('preproc',   preprocessor),
    ('clf',       LogisticRegression(
                     random_state=42,
                     max_iter=1000,
                     class_weight='balanced'
                 ))
])

if __name__ == "__main__":
    # Sólo se ejecuta cuando corro “python docker/create_pipeline.py”
    DATA_CSV = os.getenv("DATA_CSV", "docker/files/weatherAUS.csv")
    df2 = pd.read_csv(DATA_CSV)

    # Eliminar filas sin etiqueta
    df2 = df2.dropna(subset=["RainTomorrow"])

    pipeline.fit(df2, df2["RainTomorrow"])
    MODEL_PKL = os.getenv("MODEL_PKL", "docker/files/pipeline.pkl")
    joblib.dump(pipeline, MODEL_PKL)
    print(f"Pipeline entrenado y guardado en {MODEL_PKL}")