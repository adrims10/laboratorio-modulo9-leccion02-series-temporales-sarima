import pandas as pd
import numpy as np
import pickle

# Cargar el modelo y los transformers al inicio
with open('mejor_modelo_xgboost.pkl', 'rb') as f:
    model = pickle.load(f)

with open('transformer_scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('transformer_target.pkl', 'rb') as f:
    target = pickle.load(f)

with open('transformer_one.pkl', 'rb') as f:
    one = pickle.load(f)

# Variables que necesitan transformación one-hot
variables_one = ['MaritalStatus', 'JobRole', 'BusinessTravel', 'Department', 'EducationField']

# Función para realizar la predicción
def predecir_attrition(data):
    """
    Realiza la predicción de Attrition y devuelve las probabilidades.
    
    Parámetros:
    - data (dict): Diccionario con las características del empleado.
    
    Retorna:
    - prediction: Clase predicha (0 o 1).
    - prob: Probabilidades para cada clase ([probabilidad de 0, probabilidad de 1]).
    """
    # Convertir el diccionario en un DataFrame
    df_pred = pd.DataFrame(data, index=[0])
    
    # Identificar columnas numéricas
    col_numericas = df_pred.select_dtypes(include=np.number).columns
    
    # Escalar las variables numéricas
    df_pred[col_numericas] = scaler.transform(df_pred[col_numericas])
    
    # Transformar las variables categóricas one-hot
    df_one = pd.DataFrame(one.transform(df_pred[variables_one]).toarray(), 
                          columns=one.get_feature_names_out())
    df_pred = pd.concat([df_pred, df_one], axis=1)
    df_pred.drop(columns=variables_one, axis=1, inplace=True)
    
    # Transformar las variables categóricas del target
    df_pred = target.transform(df_pred)
    
    # Eliminar la columna 'Attrition' si existe
    if 'Attrition' in df_pred.columns:
        df_pred.drop(columns=['Attrition'], inplace=True)
    
    # Realizar la predicción
    prediction = model.predict(df_pred)[0]
    prob = model.predict_proba(df_pred)[0]
    
    return prediction, prob

