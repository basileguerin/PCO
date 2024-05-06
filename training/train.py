import mysql.connector
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
from keras.models import Model
from keras.layers import Input, GRU, Concatenate, Dense, Flatten
from keras.callbacks import ModelCheckpoint

# Connexion à la database
conn = mysql.connector.connect(
    host='localhost',
    port='3307',
    user='root',
    password='example',
    database='PCO'
)

# Création du curseur
cursor = conn.cursor()

# Récupération des données hospitalières
query_urg = "SELECT * FROM URG_AD;"
df_urg = pd.read_sql(query_urg, conn, index_col='date', parse_dates=['date'])

# Récupération des données externes
query_ext = "SELECT * FROM external;"
df_externe = pd.read_sql(query_ext, conn, index_col='date', parse_dates=['date'])

# Fermeture de la connexion
cursor.close()
conn.close()

# Fusion des dataframes
df = pd.merge(df_urg, df_externe, left_index=True, right_index=True, how='inner')

# Scaling des données
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)

# Fonction pour créer des séquences d'entraînement
def create_sequence(df, sequence_length):
    X = []
    y = []

    for i in range(len(df) - sequence_length):
        X.append(df.iloc[i:i+sequence_length])
        y.append(df.iloc[i+sequence_length, df.columns.get_loc('value')])
    
    return np.array(X), np.array(y)

# Création des séquences
sequence_length = 14
X, y = create_sequence(df_scaled, sequence_length)

# Séparation des données
input_1 = X[:, :, 1:]  # Toutes les colonnes sauf 'value'
input_2 = X[:, :, 0].reshape(-1, sequence_length, 1)  # Uniquement la colonne 'value'

# Création du modèle
input_layer_1 = Input(shape=(sequence_length, input_1.shape[2]))
input_layer_2 = Input(shape=(sequence_length, 1))

gru_layer_1 = GRU(units=64, return_sequences=True)(input_layer_1)
gru_layer_2 = GRU(units=64, return_sequences=True)(input_layer_2)

merged_layer = Concatenate(axis=-1)([gru_layer_1, gru_layer_2])

flattened_layer = Flatten()(merged_layer)

output_layer = Dense(1)(flattened_layer)

model = Model(inputs=[input_layer_1, input_layer_2], outputs=output_layer)

model.compile(optimizer='adam', loss='mse')

# Sauvegarde du modèle après chaque epoch
checkpoint = ModelCheckpoint('./training/best_model.h5', save_best_only=True, monitor='loss', mode='min')

# Entraînement du modèle
model.fit([input_1, input_2], y, epochs=1000, batch_size=32, callbacks=[checkpoint])

# Sauvegarde du scaler
pickle.dump(scaler, open('./training/scaler.pkl', 'wb'))