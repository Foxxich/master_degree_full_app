import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time


class Predictor:
    @staticmethod
    def predict(X_test, ids, model):
        # Logowanie informacji o rozpoczęciu predykcji
        logging.info("Starting predictions on test data...")

        # Zmierz czas rozpoczęcia predykcji
        start_time = time.time()

        # Predykcja etykiet na danych testowych
        y_pred = model.predict(X_test)

        # Predykcja prawdopodobieństw klas
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Obliczenie czasu trwania predykcji
        prediction_time = time.time() - start_time

        # Stworzenie DataFrame z identyfikatorami i przewidzianymi etykietami oraz prawdopodobieństwami
        results = pd.DataFrame({'id': ids, 'predicted_label': y_pred, 'predicted_proba': y_pred_proba})

        # Logowanie informacji o zakończeniu predykcji
        logging.info(f"Predictions completed in {prediction_time:.4f} seconds")
        return results, prediction_time

    @staticmethod
    def calculate_metrics(results, submit_data):
        # Logowanie informacji o obliczaniu metryk
        logging.info("Calculating metrics for predictions...")

        # Połączenie przewidzianych wyników z danymi referencyjnymi na podstawie identyfikatorów
        merged = results.merge(submit_data, on='id')
        y_true = merged['label']
        y_pred = merged['predicted_label']
        y_proba = merged['predicted_proba']

        # Obliczenie różnych metryk
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)

        # Zwrócenie obliczonych metryk jako słownika
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

        # Logowanie informacji o obliczonych metrykach
        logging.info(f"Metrics calculated: {metrics}")
        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path, algorithm):
        # Logowanie informacji o rysowaniu macierzy pomyłek
        logging.info("Plotting confusion matrix...")

        # Obliczenie macierzy pomyłek
        cm = confusion_matrix(y_true, y_pred)

        # Rysowanie macierzy pomyłek za pomocą seaborn
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {algorithm}')

        # Zapisanie wykresu do pliku
        plt.savefig(os.path.join(save_path, f'confusion_matrix_{algorithm}.png'))
        plt.close()

    @staticmethod
    def plot_precision_recall_curve(y_true, y_proba, save_path, algorithm):
        # Logowanie informacji o rysowaniu krzywej precision-recall
        logging.info("Plotting precision-recall curve...")

        # Obliczenie krzywej precision-recall
        precision, recall, _ = precision_recall_curve(y_true, y_proba)

        # Rysowanie krzywej precision-recall
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {algorithm}')

        # Zapisanie wykresu do pliku
        plt.savefig(os.path.join(save_path, f'precision_recall_curve_{algorithm}.png'))
        plt.close()

    @staticmethod
    def plot_roc_curve(y_true, y_proba, save_path, algorithm):
        # Logowanie informacji o rysowaniu krzywej ROC
        logging.info("Plotting ROC curve...")

        # Obliczenie krzywej ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)

        # Rysowanie krzywej ROC
        plt.plot(fpr, tpr, marker='.')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {algorithm} (AUC = {roc_auc:.2f})')

        # Zapisanie wykresu do pliku
        plt.savefig(os.path.join(save_path, f'roc_curve_{algorithm}.png'))
        plt.close()

    @staticmethod
    def analyze_simulation_impact(results, simulation_results, submit_data):
        """Analizuje wpływ wyników symulacji na predykcje modelu."""
        logging.info("Analyzing simulation impact on predictions...")

        # Połączenie wyników symulacji z wynikami predykcji
        merged = results.merge(submit_data, on='id')

        # Ensure that the user_id exists in simulation_results
        def get_simulated_influence(user_id):
            return len(simulation_results.get(user_id, []))

        merged['simulated_influence'] = merged['id'].apply(get_simulated_influence)

        # Handle case where all simulated_influence values are zero
        if merged['simulated_influence'].sum() == 0:
            logging.warning("All simulated_influence values are zero, skipping weighted accuracy calculation.")
            accuracy_with_simulation = accuracy_score(merged['label'], merged['predicted_label'])
        else:
            accuracy_with_simulation = accuracy_score(merged['label'], merged['predicted_label'],
                                                      sample_weight=merged['simulated_influence'])

        logging.info(f"Accuracy considering simulation influence: {accuracy_with_simulation:.4f}")
        return accuracy_with_simulation

