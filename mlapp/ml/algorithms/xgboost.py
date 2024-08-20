from xgboost import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline


def get_model(max_features=5000, n_estimators=100):
    # Inicjalizacja wektoryzatora TF-IDF z maksymalną liczbą cech równą max_features
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Inicjalizacja klasyfikatora XGBoost
    # n_estimators=100 oznacza, że model XGBoost będzie miał 100 estymatorów
    # use_label_encoder=False i eval_metric='logloss' są domyślnymi ustawieniami w celu uniknięcia ostrzeżeń
    model = XGBClassifier(n_estimators=n_estimators, use_label_encoder=False, eval_metric='logloss')

    # Pipeline łączy wektoryzację i model w jedną całość
    pipeline = Pipeline([
        ('vectorizer', vectorizer),  # Krok 1: Wektoryzacja tekstu
        ('classifier', model)  # Krok 2: Klasyfikacja przy użyciu XGBoost
    ])

    # Zwrócenie pełnego pipeline'u
    return pipeline
