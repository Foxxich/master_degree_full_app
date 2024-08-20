import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        # Warstwa w pełni połączona (linear layer), która przekształca wejście o wymiarze input_dim do przestrzeni o wymiarze 100
        self.layer1 = nn.Linear(input_dim, 100)

        # Kolejna warstwa w pełni połączona, zmniejszająca wymiar do 50
        self.layer2 = nn.Linear(100, 50)

        # Ostatnia warstwa w pełni połączona, która mapuje do przestrzeni wyjściowej z 2 klasami (np. 0 lub 1)
        self.layer3 = nn.Linear(50, 2)

    def forward(self, x):
        # Zastosowanie funkcji aktywacji ReLU po każdej warstwie
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        # Ostatnia warstwa nie ma aktywacji, ponieważ jest to warstwa wyjściowa
        x = self.layer3(x)
        return x


class PyTorchClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, model, epochs=10, batch_size=32, learning_rate=0.001):
        # Inicjalizacja modelu, liczby epok, wielkości partii danych oraz szybkości uczenia
        self.model = model
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Wektoryzator TF-IDF, który przekształca teksty w wektory cech
        self.vectorizer = TfidfVectorizer(max_features=5000)

    def fit(self, X, y):
        # Przekształcenie danych treningowych na wektory TF-IDF
        X_tfidf = self.vectorizer.fit_transform(X).toarray()

        # Konwersja danych wejściowych do tensora PyTorch
        X_tfidf_tensor = torch.tensor(X_tfidf, dtype=torch.float32)
        y_tensor = torch.tensor(y.values, dtype=torch.long)

        # Inicjalizacja optymalizatora Adam i funkcji straty
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Proces treningu modelu
        self.model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()  # Wyzerowanie gradientów z poprzedniej iteracji
            outputs = self.model(X_tfidf_tensor)  # Przepuszczenie danych przez model
            loss = criterion(outputs, y_tensor)  # Obliczenie straty
            loss.backward()  # Obliczenie gradientów
            optimizer.step()  # Aktualizacja wag modelu

        return self

    def predict_proba(self, X):
        # Przekształcenie danych testowych na wektory TF-IDF
        X_tfidf = self.vectorizer.transform(X).toarray()
        X_tfidf_tensor = torch.tensor(X_tfidf, dtype=torch.float32)

        # Predykcja prawdopodobieństw klas przez model
        self.model.eval()
        with torch.no_grad():  # Wyłączenie obliczania gradientów podczas predykcji
            outputs = self.model(X_tfidf_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().tolist()
            probabilities = np.array(probabilities)
        return probabilities

    def predict(self, X):
        # Predykcja ostatecznych klas na podstawie prawdopodobieństw
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def get_model(max_features=5000, epochs=10, batch_size=32, learning_rate=0.001):
    # Tworzenie modelu sieci neuronowej SimpleNN z podanym wymiarem wejściowym
    model = SimpleNN(input_dim=max_features)

    # Zwrócenie obiektu PyTorchClassifier, który owija model SimpleNN w interfejs zgodny ze scikit-learn
    return PyTorchClassifier(model, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate)
