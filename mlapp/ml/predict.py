import matplotlib
import pandas as pd
import logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend

class Predictor:
    @staticmethod
    def predict(X_test, ids, model):
        logging.info("Starting predictions on test data...")
        start_time = time.time()
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        prediction_time = time.time() - start_time
        results = pd.DataFrame({'id': ids, 'predicted_label': y_pred, 'predicted_proba': y_pred_proba})
        logging.info(f"Predictions completed in {prediction_time:.4f} seconds")
        return results, prediction_time

    @staticmethod
    def calculate_metrics(results, submit_data):
        logging.info("Calculating metrics for predictions...")
        merged = results.merge(submit_data, on='id')
        y_true = merged['label']
        y_pred = merged['predicted_label']
        y_proba = merged['predicted_proba']

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        roc_auc = roc_auc_score(y_true, y_proba)

        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }

        logging.info(f"Metrics calculated: {metrics}")
        return metrics

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred, save_path, algorithm):
        logging.info("Plotting confusion matrix...")
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {algorithm}')
        plt.savefig(os.path.join(save_path, f'confusion_matrix_{algorithm}.png'))
        plt.close()

    @staticmethod
    def plot_precision_recall_curve(y_true, y_proba, save_path, algorithm):
        logging.info("Plotting precision-recall curve...")
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {algorithm}')
        plt.savefig(os.path.join(save_path, f'precision_recall_curve_{algorithm}.png'))
        plt.close()

    @staticmethod
    def plot_roc_curve(y_true, y_proba, save_path, algorithm):
        logging.info("Plotting ROC curve...")
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, marker='.')
        plt.plot([0, 1], [0, 1], linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {algorithm} (AUC = {roc_auc:.2f})')
        plt.savefig(os.path.join(save_path, f'roc_curve_{algorithm}.png'))
        plt.close()
