from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
import pickle
import logging
import numpy as np

from .data_loader import DataLoader

class FakeNewsModel:
    def __init__(self, algorithm='voting'):
        self.vectorizer = TfidfVectorizer(max_features=5000)

        # Define individual models
        self.rf = RandomForestClassifier(n_estimators=100)
        self.ada = AdaBoostClassifier(n_estimators=100, algorithm='SAMME')
        self.gbc = GradientBoostingClassifier(n_estimators=100)
        self.xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        self.lr = LogisticRegression(max_iter=1000)

        # Choose the ensemble method
        if algorithm == 'random_forest':
            self.model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.rf)
            ])
        elif algorithm == 'adaboost':
            self.model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.ada)
            ])
        elif algorithm == 'gradient_boosting':
            self.model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.gbc)
            ])
        elif algorithm == 'xgboost':
            self.model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.xgb)
            ])
        elif algorithm == 'logistic_regression':
            self.model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.lr)
            ])
        elif algorithm == 'voting':
            self.ensemble = VotingClassifier(estimators=[
                ('rf', self.rf),
                ('ada', self.ada),
                ('gbc', self.gbc),
                ('xgb', self.xgb),
                ('lr', self.lr)
            ], voting='soft')
            self.model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.ensemble)
            ])
        elif algorithm == 'custom_weighted_voting':
            # Custom weighted voting ensemble
            weights = self.calculate_weights()
            self.ensemble = VotingClassifier(estimators=[
                ('rf', self.rf),
                ('ada', self.ada),
                ('gbc', self.gbc),
                ('xgb', self.xgb),
                ('lr', self.lr)
            ], voting='soft', weights=weights)
            self.model = Pipeline([
                ('vectorizer', self.vectorizer),
                ('classifier', self.ensemble)
            ])
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self.algorithm = algorithm

    def calculate_weights(self):
        # Custom method to calculate weights for each model based on validation performance
        X_train, y_train = DataLoader.load_train_data('data/train.csv')
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        models = [self.rf, self.ada, self.gbc, self.xgb, self.lr]
        weights = []
        for model in models:
            model.fit(X_train_tfidf, y_train)
            accuracy = model.score(X_train_tfidf, y_train)
            weights.append(accuracy)
        return weights

    def train(self, X_train, y_train):
        logging.info(f"Starting model training with {self.algorithm}...")
        self.model.fit(X_train, y_train)
        with open(f'model_{self.algorithm}.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        logging.info(f"Model training completed and saved as model_{self.algorithm}.pkl with {self.algorithm}")

    @staticmethod
    def load_model(algorithm):
        logging.info(f"Loading trained model from model_{algorithm}.pkl")
        with open(f'model_{algorithm}.pkl', 'rb') as f:
            model = pickle.load(f)
        logging.info("Model loaded")
        return model
