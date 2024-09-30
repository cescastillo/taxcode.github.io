
from openpyxl import reader, load_workbook, Workbook
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import OrdinalEncoder
import streamlit as st
import pandas as pd
import numpy as np
import time
import requests
import io
import joblib


# Para correr el programa localmente usar streamlit run main.py
st.set_page_config(layout="centered", page_title="Tax Code Model")

st.title("Encode Tax Code")
st.subheader("Instrucciones de uso")
st.write("Paso 1: Selección del modelo del impuesto:")
st.write("En el panel lateral de la aplicación encontrarás un menú desplegable con el título Selecciona el impuesto. Aquí deberás seleccionar el tipo de impuesto para el cual deseas realizar la predicción del código. Las opciones disponibles son: 1120, 1065, 1040 y 1120S")
st.write("")
st.write("Paso 2: Cargar el archivo de Datos")
st.write("Debajo del menú de selección de impuesto, verás un botón que dice Carga el archivo. Aquí deberás subir el archivo de datos financieros. La aplicación acepta archivos en formato Excel .xlsx, csv o txt ")
st.write("Consideraciones importantes: Asegúrate de que el archivo no exceda los 10 MB, El archivo debe contener una hoja llamada Sheet1 u Hoja1, donde se encuentran los datos relevantes")
st.write("")
st.write("Paso 3: Visualización de la Predicción")
st.write("Una vez que hayas subido el archivo y seleccionado el impuesto, la aplicación procesará los datos y aplicará el modelo correspondiente. Luego de unos segundos, se mostrará en pantalla una tabla con las siguientes columnas:")
st.text("Predicted Tax Code: El código fiscal predicho.")
st.text("Account Number: El número de cuenta.")
st.text("Account Description: La descripción de la cuenta.")
st.text("Debit: La cantidad debitada.")
st.text("Credit: La cantidad acreditada.")



#DEFININEDO VARIABLES Y FUNCIONES
archivo_excel = "SUPERMERCADO TALPA 7, INC_2023_FINANCIAL STATEMENTS_TRIAL BALANCE 2023.xlsx"
max_file_size = 10 * 1024 * 1024 #10MB
logoGllv = "./imgs/GLLV Logo.png"
df_completo = None

def load_model_from_dropbox(url):
   # Descargar el archivo desde dropbox
   response = requests.get(url)
   response.raise_for_status() # Verificar que no haya errores en la solicitud
   # Cargar el modelo desde los bytes descargados
   return joblib.load(io.BytesIO(response.content))

# URLs de Dropbox con el formato dl=1 para descarga directa
dropbox_model1 = 'https://www.dropbox.com/scl/fi/56ftsf5wdlt4d47ybwu88/model_1120S_compress.pkl?rlkey=e0darh6fgtji6xqn2lzqwobpy&st=onzq2afs&dl=1'
dropbox_model2 = 'https://www.dropbox.com/scl/fi/uox9v46okb8ygapjflrdz/model_1120_compress.pkl?rlkey=euqm9e5xwoqx2qd9cjby6hya4&st=aanpo3tw&dl=1'
dropbox_model3 = 'https://www.dropbox.com/scl/fi/ejdh2ktptr31gu1f0faxo/model_1065_compress.pkl?rlkey=w3fvn41qjfsap8aioo3a4ljnt&st=4t4hyi38&dl=1'
dropbox_model4 = 'https://www.dropbox.com/scl/fi/mjwclwnbhig67ypa6e643/model_1040.pkl?rlkey=6u729vhuw4318n4rtlgqd91xj&st=0z9f83se&dl=1'

dropbox_encoder1 = 'https://www.dropbox.com/scl/fi/xwgso10vsqkzyawr3pzke/ordinal_encoder1.pkl?rlkey=xqufej7vfk50nmpakq60xql32&st=vzwxnivj&dl=1'
dropbox_encoder2 = 'https://www.dropbox.com/scl/fi/omtczdp8xf0gfxz0gf4jz/ordinal_encoder2.pkl?rlkey=jkonn4l8imv4vyemsixnesh1g&st=us28py3w&dl=1'

# Cargar los modelos y encoders desde Dropbox
model1 = load_model_from_dropbox(dropbox_model1)
model2 = load_model_from_dropbox(dropbox_model2)
model3 = load_model_from_dropbox(dropbox_model3)
model4 = load_model_from_dropbox(dropbox_model4)

#Usar para generar los modelos comprimidos
# joblib.dump(model1,'model_1120S_compress.pkl', compress=3)
# print("Modelo 1120S comprimido")


#Definir entity_tax_models como un diccionario
entity_tax_models = {
    '1120': model2, # Puedes reemplazar 'Modelo 1120S' con el modelo real o función
    '1065': model3,
    '1040': model4,
    '1120S': model1
     # Agrega más casos según sea necesario
}

ordinal_encoder1 = load_model_from_dropbox(dropbox_encoder1)
print("ordinal_encoder1 cargado desde 'ordinal_encoder1.pkl' ")

ordinal_encoder2 = load_model_from_dropbox(dropbox_encoder2)
print("ordinal_encoder2 cargado desde 'ordinal_encoder2.pkl' ")


st.sidebar.title("Por favor carga el archivo y selecciona el modelo del impuesto")

#REEMPLAZAR ESTA LECTURA DEL EXCEL POR UN UPLOADER DE STREAMLIT, PARA PROBAR CON LOS ARCHIVOS QUE SE SUBEN A TRAVES DE ESE UPLOADER
add_selectbox = st.sidebar.selectbox('Selecciona el impuesto',('1120','1065','1040','1120S'))
print(f"{add_selectbox}")

uploaded_file = st.sidebar.file_uploader("Carga el archivo", type={"xlsx","csv","txt"})


if uploaded_file is not None:
 try:
     df_completo = pd.read_excel(uploaded_file, sheet_name='Sheet1')
     print("Hoja 'Sheet1' cargada exitosamente.")
 except ValueError:
     #Si falla, intentar con la hoja 'Hoja1'
     try:
         df_completo = pd.read_excel(uploaded_file, sheet_name='Hoja1')
         print("Hoja 'Hoja1' cargada exitosamente.")
     except ValueError:
         raise ValueError("No se pudo encontrar 'Shee1' ni 'Hoja1' en el archivo Excel")
         
   

#Mostrar las columnas disponibles
if isinstance(df_completo, pd.DataFrame):
 print(df_completo.columns.to_list())



 #Seleccionar las filas que contienen los datos (a partir de la fila 1)
 df_completo = df_completo.iloc[1:].reset_index(drop=True)
 #Renombrar las columnas alos datos que tenemos
 df_completo.columns = ['Unnamed','Account Description','Debit','Unnamed_3','Credit']

 #Eliminar columnas innecesarias
 df_completo = df_completo[['Account Description','Debit','Credit']]

 df_completo['Account Number'] = df_completo['Account Description'].str.split('·').str[0]
 df_completo['Account Description'] = df_completo['Account Description'].str.split('·').str[1]

 #Define las columnas que deseas extraer
 columnas_deseadas = ['Account Number','Account Description','Debit','Credit']

 #Verificamos que las columnas existan en el archivo
 for columna in columnas_deseadas:
    if columna not in df_completo.columns:
        raise ValueError(f"La columna '{columna}' no se encuentra en el archivo de Excel.")

 #Seleccionamos las columnas deseadas
 df = df_completo[columnas_deseadas].copy()

 # Opcional: Manejo de valores faltantes
 df['Debit'] = df['Debit'].fillna(0)
 df['Credit'] = df['Credit'].fillna(0)

 #Asegurar que 'Debit' y 'Credit' sean de tipo numérico
 df['Debit'] = pd.to_numeric(df['Debit'],errors='coerce').fillna(0)
 df['Credit'] = pd.to_numeric(df['Credit'],errors='coerce').fillna(0)

 df['Entity Tax'] = f"{add_selectbox}"


 df = df.dropna(subset=['Account Number', 'Account Description'], how='all')
 new_data = df

 # Convertir las columnas de los nuevos datos a los tipos correctos
 new_data['Entity Tax'] = new_data['Entity Tax'].astype(str)
 new_data['Account Description2'] = new_data['Account Description'].astype(str)
 new_data['Account Number2'] = new_data['Account Number'].astype(str)
 new_data['Account Description2'] = new_data['Account Description2'].str.strip().str.lower()
 new_data['Account Number2'] = new_data['Account Number2'].str.strip().str.lower()


 new_data['Account Description2'] = ordinal_encoder2.transform(new_data['Account Description2'].values.reshape(-1, 1))
 new_data['Account Number2'] = ordinal_encoder1.transform(new_data['Account Number2'].values.reshape(-1, 1))


 # Predicción
 predictions = []

 for _, row in new_data.iterrows():
    entity_tax = row['Entity Tax']

    # Seleccionar el modelo correcto basado en Entity Tax
    if entity_tax in entity_tax_models:
        model = entity_tax_models[entity_tax]

        # Seleccionar las características necesarias
        X_new = row[['Account Number2', 'Account Description2']].values.reshape(1, -1)

        # Realizar la predicción
        prediction = model.predict(X_new)

        # Almacenar la predicción
        predictions.append(prediction[0])
    else:
        predictions.append(None)  # Si no hay modelo para el Entity Tax, devolver None

 # Añadir las predicciones al DataFrame
 new_data['Predicted Tax Code'] = predictions

 new_data_selected = new_data[['Predicted Tax Code','Account Number', 'Account Description', 'Debit', 'Credit']]

 st.write(f"## Prediccion de Tax Code {add_selectbox}")
 st.dataframe(new_data_selected)
 st.text("Para obtener soporte adicional, comunícate con el desarrollador de la aplicación.")


st.logo(logoGllv)




