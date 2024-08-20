import logging
import os
import time
import pandas as pd
from .data_loader import DataLoader
from .model import FakeNewsModel
from .predict import Predictor


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Load the training data
    X_train, y_train = DataLoader.load_train_data('data/train.csv')

    # Prompt the user for the algorithm choice
    print("What algorithm would you like to use?")
    print("1. Random Forest")
    print("2. AdaBoost")
    print("3. Gradient Boosting")
    print("4. XGBoost")
    print("5. Logistic Regression")
    print("6. Voting")
    print("7. Custom Weighted Voting")
    choice = input("Enter the number of your choice: ")

    algorithms = {
        '1': 'random_forest',
        '2': 'adaboost',
        '3': 'gradient_boosting',
        '4': 'xgboost',
        '5': 'logistic_regression',
        '6': 'voting',
        '7': 'custom_weighted_voting'
    }

    algorithm = algorithms.get(choice, 'voting')
    logging.info(f"Starting fake news detection process with {algorithm}...")

    # Ensure the plots directory exists
    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)

    # Train the model
    model = FakeNewsModel(algorithm)
    start_time = time.time()
    model.train(X_train, y_train)
    training_time = time.time() - start_time

    # Load the test data
    X_test, ids = DataLoader.load_test_data('data/test.csv')

    # Load the trained model
    trained_model = FakeNewsModel.load_model(algorithm)

    # Predict on the test data
    predictions, prediction_time = Predictor.predict(X_test, ids, trained_model)

    # Load the submit data for comparison
    submit_data = DataLoader.load_submit_data('data/submit.csv')

    # Calculate metrics
    metrics = Predictor.calculate_metrics(predictions, submit_data)
    metrics['training_time'] = training_time
    metrics['prediction_time'] = prediction_time

    # Save metrics to CSV
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{algorithm}_stats.csv', index=False)
    logging.info(f"Metrics saved to {algorithm}_stats.csv")

    # Save comparison results to results.csv
    predictions.to_csv(f'results_{algorithm}.csv', index=False)
    logging.info(f"Results saved to results_{algorithm}.csv")

    # Extract true labels for comparison
    y_true = submit_data['label']
    y_pred = predictions['predicted_label']
    y_proba = predictions['predicted_proba']

    # Plot confusion matrix
    Predictor.plot_confusion_matrix(y_true, y_pred, plot_dir, algorithm)

    # Plot precision-recall curve

