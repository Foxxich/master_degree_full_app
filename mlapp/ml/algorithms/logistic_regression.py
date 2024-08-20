from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def get_model(max_features=5000):
    # Inicjalizacja wektoryzatora TF-IDF z maksymalną liczbą cech równą max_features
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Inicjalizacja klasyfikatora regresji logistycznej
    # max_iter=1000 oznacza, że algorytm będzie próbował znaleźć rozwiązanie w maksymalnie 1000 iteracjach
    model = LogisticRegression(max_iter=1000)

    # Pipeline łączy wektoryzację i model w jedną całość
    pipeline = Pipeline([
        ('vectorizer', vectorizer),  # Krok 1: Wektoryzacja tekstu
        ('classifier', model)  # Krok 2: Klasyfikacja przy użyciu regresji logistycznej
    ])

    # Zwrócenie pełnego pipeline'u
    return pipeline
