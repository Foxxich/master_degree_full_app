import pandas as pd
import logging

class DataLoader:
    @staticmethod
    def load_train_data(file_path):
        logging.info(f"Loading training data from {file_path}")
        train_data = pd.read_csv(file_path)
        train_data['combined'] = train_data['title'].fillna('') + " " + train_data['text'].fillna('')
        X_train = train_data['combined']
        y_train = train_data['label']
        logging.info(f"Training data loaded: {len(train_data)} records")
        return X_train, y_train

    @staticmethod
    def load_test_data(file_path):
        logging.info(f"Loading test data from {file_path}")
        test_data = pd.read_csv(file_path)
        test_data['combined'] = test_data['title'].fillna('') + " " + test_data['text'].fillna('')
        X_test = test_data['combined']
        ids = test_data['id']
        logging.info(f"Test data loaded: {len(test_data)} records")
        return X_test, ids

    @staticmethod
    def load_submit_data(file_path):
        logging.info(f"Loading submit data from {file_path}")
        submit_data = pd.read_csv(file_path)
        logging.info(f"Submit data loaded: {len(submit_data)} records")
        return submit_data
