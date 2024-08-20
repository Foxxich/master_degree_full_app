from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class HybridEnsemble(BaseEstimator, ClassifierMixin):
    def __init__(self, models, halting_threshold=0.95):
        # Inicjalizacja hybrydowego modelu ensemble z podanymi modelami bazowymi
        # halting_threshold definiuje próg, po którego przekroczeniu model kończy sekwencyjne przetwarzanie
        self.models = models
        self.halting_threshold = halting_threshold

        # Wektoryzator TF-IDF, który przekształca teksty w wektory cech
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def fit(self, X, y):
        # Przekształcenie danych treningowych na wektory TF-IDF
        X_tfidf = self.vectorizer.fit_transform(X)

        # Trenowanie każdego modelu bazowego z osobna
        for model in self.models:
            model.fit(X_tfidf, y)
        return self

    def predict_proba(self, X):
        # Przekształcenie danych testowych na wektory TF-IDF
        X_tfidf = self.vectorizer.transform(X)

        # Inicjalizacja tablicy do przechowywania średnich prawdopodobieństw
        avg_proba = np.zeros((X_tfidf.shape[0], 2))

        # Zsumowanie prawdopodobieństw z każdego modelu bazowego
        for model in self.models:
            avg_proba += model.predict_proba(X_tfidf)

        # Uśrednienie prawdopodobieństw przez liczbę modeli
        avg_proba /= len(self.models)
        return avg_proba

    def predict(self, X):
        # Predykcja ostatecznych klas na podstawie uśrednionych prawdopodobieństw
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def sequential_predict(self, X):
        # Sequentialna predykcja, w której modele bazowe są używane po kolei,
        # aż do momentu przekroczenia zadanego progu halting_threshold
        X_tfidf = self.vectorizer.transform(X)
        avg_proba = np.zeros((X_tfidf.shape[0], 2))
        for i, model in enumerate(self.models):
            avg_proba += model.predict_proba(X_tfidf)
            avg_proba /= (i + 1)

            # Przerwanie procesu, jeśli średnie maksymalne prawdopodobieństwo przekracza próg
            if np.max(avg_proba, axis=1).mean() > self.halting_threshold:
                break
        return np.argmax(avg_proba, axis=1)


def get_model(max_features=5000, n_estimators=100):
    # Inicjalizacja wektoryzatora TF-IDF z podaną maksymalną liczbą cech
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Lista modeli bazowych, z których każdy jest umieszczony w pipeline
    models = [
        Pipeline([('classifier', SVC(probability=True))]),
        # Support Vector Classifier z włączonym zwracaniem prawdopodobieństw
        Pipeline([('classifier', MLPClassifier(max_iter=1000))])
        # MLPClassifier (Wielowarstwowa Sieć Neuronowa) z maksymalną liczbą iteracji 1000
    ]

    # Zwrócenie hybrydowego modelu ensemble
    return HybridEnsemble(models)
