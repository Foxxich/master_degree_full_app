import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, nhead, num_encoder_layers, dim_feedforward, num_classes):
        super(TransformerClassifier, self).__init__()
        # Warstwa osadzająca (embedding), która przekształca dane wejściowe do przestrzeni o wyższym wymiarze
        self.embedding = nn.Linear(input_dim, dim_feedforward)

        # TransformerEncoder składający się z kilku warstw enkodera Transformer, które będą przetwarzać sekwencje danych
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim_feedforward, nhead=nhead),
            num_layers=num_encoder_layers
        )

        # Warstwa w pełni połączona (fully connected), która na podstawie wyjścia z transformera przewiduje klasy
        self.fc = nn.Linear(dim_feedforward, num_classes)

    def forward(self, x):
        # Przekształcenie danych wejściowych za pomocą warstwy osadzającej
        x = self.embedding(x)

        # Dodanie wymiaru, aby symulować długość sekwencji równą 1
        x = x.unsqueeze(1)  # Teraz kształt x to (batch_size, 1, dim_feedforward)

        # Zmiana kolejności wymiarów, aby dopasować do wymagań transformera (seq_len, batch_size, embedding_dim)
        x = x.permute(1, 0, 2)

        # Przekazanie danych przez enkoder transformera
        x = self.transformer(x)

        # Uśrednienie wyników po długości sekwencji, aby uzyskać jedno wyjście na próbkę
        x = x.mean(dim=0)  # Teraz kształt x to (batch_size, dim_feedforward)

        # Przejście przez warstwę w pełni połączoną w celu uzyskania ostatecznych wyników klasyfikacji
        x = self.fc(x)
        return x


class PyTorchTransformerClassifier(BaseEstimator, ClassifierMixin):
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
            optimizer.zero_grad()
            outputs = self.model(X_tfidf_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()  # Obliczenie gradientów
            optimizer.step()  # Aktualizacja wag modelu

        return self

    def predict_proba(self, X):
        # Przekształcenie danych testowych na wektory TF-IDF
        X_tfidf = self.vectorizer.transform(X).toarray()
        X_tfidf_tensor = torch.tensor(X_tfidf, dtype=torch.float32)

        # Predykcja prawdopodobieństw klas przez model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tfidf_tensor)
            probabilities = torch.softmax(outputs, dim=1).cpu().tolist()
            probabilities = np.array(probabilities)
        return probabilities

    def predict(self, X):
        # Predykcja ostatecznych klas na podstawie prawdopodobieństw
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)


def get_model(max_features=5000, nhead=4, num_encoder_layers=3, dim_feedforward=512):
    # Inicjalizacja klasyfikatora Transformer z podanymi parametrami
    model = TransformerClassifier(
        input_dim=max_features,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        num_classes=2
    )
    # Zwrócenie obiektu PyTorchTransformerClassifier, który owija model w interfejs zgodny ze scikit-learn
    return PyTorchTransformerClassifier(model)
