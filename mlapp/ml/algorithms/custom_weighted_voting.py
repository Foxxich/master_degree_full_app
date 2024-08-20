from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def get_model(max_features=5000, n_estimators=100):
    # Inicjalizacja wektoryzatora TF-IDF z maksymalną liczbą cech równą max_features
    vectorizer = TfidfVectorizer(max_features=max_features)

    # Inicjalizacja trzech różnych klasyfikatorów bazowych
    rf = RandomForestClassifier(n_estimators=n_estimators)  # Las losowy
    ada = AdaBoostClassifier(n_estimators=n_estimators)     # AdaBoost
    gbc = GradientBoostingClassifier(n_estimators=n_estimators)  # Gradient Boosting

    # Stworzenie klasyfikatora zbiorczego VotingClassifier z trzema estymatorami bazowymi
    # Ustawienie wag na [2, 1, 2] oznacza, że większa waga jest przypisana do Random Forest i Gradient Boosting
    ensemble = VotingClassifier(estimators=[
        ('rf', rf),
        ('ada', ada),
        ('gbc', gbc)
    ], voting='soft', weights=[2, 1, 2])

    # Pipeline łączy wektoryzację i model zbiorczy w jedną całość
    pipeline = Pipeline([
        ('vectorizer', vectorizer),  # Krok 1: Wektoryzacja tekstu
        ('classifier', ensemble)     # Krok 2: Klasyfikacja przy użyciu VotingClassifier
    ])

    # Zwrócenie pełnego pipeline'u
    return pipeline
