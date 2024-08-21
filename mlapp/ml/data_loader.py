import pandas as pd
import logging
import random  # Dodaj ten import na górze pliku

# Definicja zainteresowań
interests = ["politics", "technology", "health", "entertainment", "sports"]

def add_interests_column(df):
    df['interests'] = [random.choice(interests) for _ in range(len(df))]
    return df

class DataLoader:
    @staticmethod
    def load_train_data(file_path):
        # Wczytanie danych treningowych z pliku CSV
        train_data = pd.read_csv(file_path)

        # Dodanie kolumny 'interests'
        train_data = add_interests_column(train_data)

        # Łączenie kolumn "title" i "text" w jedną kolumnę "combined"
        train_data['combined'] = train_data['title'].fillna('') + " " + train_data['text'].fillna('')

        X_train = train_data['combined']
        y_train = train_data['label']
        return X_train, y_train

    @staticmethod
    def load_test_data(file_path):
        # Wczytanie danych testowych z pliku CSV
        test_data = pd.read_csv(file_path)

        # Dodanie kolumny 'interests'
        test_data = add_interests_column(test_data)

        # Łączenie kolumn "title" i "text" w jedną kolumnę "combined"
        test_data['combined'] = test_data['title'].fillna('') + " " + test_data['text'].fillna('')

        X_test = test_data['combined']
        ids = test_data['id']
        return X_test, ids

    @staticmethod
    def load_submit_data(file_path):
        # Logowanie informacji o ładowaniu danych do submitowania wyników
        logging.info(f"Loading submit data from {file_path}")

        # Wczytanie danych submit z pliku CSV
        submit_data = pd.read_csv(file_path)

        # Logowanie informacji o liczbie rekordów
        logging.info(f"Submit data loaded: {len(submit_data)} records")
        return submit_data
