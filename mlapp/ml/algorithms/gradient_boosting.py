from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def get_model(max_features=5000, n_estimators=100):
    # Inicjalizacja wektoryzatora TF-IDF z maksymalną liczbą cech równą max_features
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Inicjalizacja klasyfikatora Gradient Boosting
    # n_estimators=100 oznacza, że model będzie uczył się przy użyciu 100 estymatorów bazowych
    model = GradientBoostingClassifier(n_estimators=n_estimators)

    # Pipeline łączy wektoryzację i model w jedną całość
    pipeline = Pipeline([
        ('vectorizer', vectorizer),  # Krok 1: Wektoryzacja tekstu
        ('classifier', model)  # Krok 2: Klasyfikacja przy użyciu Gradient Boosting
    ])

    # Zwrócenie pełnego pipeline'u
    return pipeline
