import streamlit as st
import mysql.connector
import pandas as pd
import numpy as np
import pickle
from keras.models import load_model
import bcrypt
import json

# Configuration de la page Streamlit
st.set_page_config(page_title='ER Admissions Forecasting', page_icon="📈", layout='wide')

# Connexion à la base de données
def get_db_connection():
    return mysql.connector.connect(
        host='localhost',
        port='3307',
        user='root',
        password='example',
        database='PCO'
    )

# Ajouter un nouvel utilisateur à la base de données
def add_new_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    try:
        cursor.execute("INSERT INTO users (username, hashed_password) VALUES (%s, %s)", (username, hashed_password))
        conn.commit()
        return True
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
        return False
    finally:
        cursor.close()
        conn.close()

# Vérifier les identifiants de l'utilisateur
def check_user(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT hashed_password FROM users WHERE username = %s", (username,))
    user_record = cursor.fetchone()
    cursor.close()
    conn.close()
    if user_record and bcrypt.checkpw(password.encode('utf-8'), user_record[0].encode('utf-8')):
        return True
    return False

# Sauvergarder les prédictions en base de données
def save_predictions(user_id, predictions):
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        prediction_date = pd.to_datetime('today').strftime('%Y-%m-%d')
        predictions_json = json.dumps(predictions)
        cursor.execute("INSERT INTO predictions (user_id, prediction_date, predicted_values) VALUES (%s, %s, %s)",
                       (user_id, prediction_date, predictions_json))
        conn.commit()
    except mysql.connector.Error as err:
        print("Something went wrong: {}".format(err))
    finally:
        cursor.close()
        conn.close()

def fetch_user_id(username):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM users WHERE username = %s", (username,))
    user_id = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    return user_id

# Interface de connexion et d'inscription
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if st.session_state['logged_in']:
    if st.button('Logout'):
        st.session_state['logged_in'] = False
        st.experimental_rerun()
    else:
        # Chargement du modèle
        model = load_model('../training/best_model.h5')

        # Récupération et préparation des données
        conn = get_db_connection()
        cursor = conn.cursor()

        # Données hospitalières
        cursor.execute("SELECT * FROM URG_AD;")
        df_urg = pd.DataFrame(cursor.fetchall(), columns=['date', 'value'])
        df_urg.set_index('date', inplace=True)

        # Données externes
        cursor.execute("SELECT * FROM external;")
        df_externe = pd.DataFrame(cursor.fetchall(), columns=['date', 'weekday_0', 'weekday_1', 'weekday_2', 'weekday_3', 'weekday_4', 'weekday_5', 
                                                             'weekday_6', 'month_1', 'month_2', 'month_3', 'month_4', 'month_5', 'month_6', 'month_7',
                                                             'month_8', 'month_9', 'month_10', 'month_11', 'month_12', 'ferie', 'vacances', 'tavg',
                                                             'tmin', 'tmax', 'prcp', 'snow', 'wspd', 'diarrhea', 'ira', 'varicelle'])
        df_externe.set_index('date', inplace=True)

        cursor.close()
        conn.close()

        df = pd.concat([df_urg, df_externe], axis=1)
        scaler = pickle.load(open('../training/scaler.pkl', 'rb'))
        df_scaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        look_back = 14
        X1 = df_scaled.drop('value', axis=1).tail(14).values
        X2 = df_scaled['value'].tail(14).values
        X1 = np.expand_dims(X1, axis=0)
        X2 = np.expand_dims(X2, axis=0)

        # Interface de visualisation et de prédiction
        st.title('Forecasting with RNN')
        period_to_visualize = st.selectbox('Select the period to visualize:', ['Entire Time Serie', 'Last Month', 'Last Year'])

        if period_to_visualize == 'Last Month':
            df_display = df['value'].tail(30)
        elif period_to_visualize == 'Last Year':
            df_display = df['value'].tail(365)
        else:
            df_display = df['value']

        st.subheader('Time Serie:')
        st.line_chart(df_display)

        if st.button('Predict the next 7 days'):
            predictions = []
            result_dict = {}
            for i in range(7):
                current_pred = model.predict([X1, X2])
                predictions.append(current_pred.flatten()[0])
                new_X2 = np.roll(X2, -1, axis=1)
                new_X2[0, -1] = current_pred
                new_X1 = np.roll(X1, -1, axis=1)
                X1, X2 = new_X1, new_X2

            temp_array = np.zeros((len(predictions), df_scaled.shape[1]))
            index_of_value = df.columns.get_loc("value")
            temp_array[:, index_of_value] = np.array(predictions).reshape(-1)
            predictions = scaler.inverse_transform(temp_array)
            predictions = predictions[:, index_of_value]

            for i, pred in enumerate(predictions):
                result_dict[(pd.to_datetime('today') + pd.DateOffset(days=i+1)).strftime('%Y-%m-%d')] = pred

            save_predictions(st.session_state['user_id'], result_dict)

            st.subheader('Model predictions :')
            st.json(result_dict)


else:
    with st.sidebar:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            if check_user(username, password):
                st.session_state['logged_in'] = True
                st.session_state['username'] = username
                st.session_state['user_id'] = fetch_user_id(username)
                st.experimental_rerun()
            else:
                st.error("Incorrect username or password")
        
        st.write("Or")
        new_username = st.text_input("New username", key="new_user")
        new_password = st.text_input("New password", type="password", key="new_pwd")
        if st.button("Register"):
            if add_new_user(new_username, new_password):
                st.success("User registered successfully. Please login.")
            else:
                st.error("Registration failed. User may already exist.")
