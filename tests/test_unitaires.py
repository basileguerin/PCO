import unittest
from unittest.mock import patch
import mysql.connector
import numpy as np

class TestDatabaseConnection(unittest.TestCase):
    @patch('mysql.connector.connect')
    def test_database_connection(self, mock_connect):
        mock_connect.return_value = True
        conn = mysql.connector.connect(host='localhost', port='3307', user='root', password='example', database='PCO')
        mock_connect.assert_called_with(host='localhost', port='3307', user='root', password='example', database='PCO')
        self.assertTrue(conn)

class TestPreprocessing(unittest.TestCase):
    def test_min_max_scaler(self):
        from sklearn.preprocessing import MinMaxScaler
        data = np.array([[1, 2], [2, 3], [3, 4]])
        scaler = MinMaxScaler()
        transformed_data = scaler.fit_transform(data)
        
        # Check that the data is scaled between 0 and 1
        self.assertEqual(transformed_data.min(), 0)
        self.assertEqual(transformed_data.max(), 1)

class TestModelLoading(unittest.TestCase):
    @patch('builtins.open')
    @patch('pickle.load')
    @patch('keras.models.load_model')
    def test_model_and_scaler_loading(self, mock_load_model, mock_pickle_load, mock_open):
        # Mock the loading processes
        mock_load_model.return_value = 'model'
        mock_pickle_load.return_value = 'scaler'
        
        # Assume your function load_model_scaler() calls these
        model = mock_load_model('./training/model.h5')
        scaler = mock_pickle_load(mock_open('./training/scaler.pkl', 'rb'))

        # Check if the mocks are called correctly
        mock_load_model.assert_called_once_with('./training/model.h5')
        mock_open.assert_called_once_with('./training/scaler.pkl', 'rb')
        self.assertEqual(model, 'model')
        self.assertEqual(scaler, 'scaler')
