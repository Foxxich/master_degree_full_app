import logging
import os
import importlib
import pandas as pd
from data_loader import DataLoader
from predict import Predictor
import matplotlib.pyplot as plt
import warnings
import time
from social_network import create_dynamic_virtual_users, create_social_network, simulate_information_spread, \
    plot_simulation_results, plot_article_read_share_frequency, plot_daily_infections, plot_cumulative_infections

# Ignoring all warnings
warnings.filterwarnings("ignore")

def run_experiment(algorithm_name, dataset_path, vectorizer_params=None, model_params=None):
    module = importlib.import_module(f'algorithms.{algorithm_name}')

    # Use default parameters if not provided
    vectorizer_params = vectorizer_params or {}
    model_params = model_params or {}

    model = module.get_model(**vectorizer_params, **model_params)

    logging.info(f"Starting fake news detection process using {algorithm_name}...")

    # Load training data from the chosen dataset
    X_train, y_train = DataLoader.load_train_data(os.path.join(dataset_path, 'train.csv'))

    # Simulate social network
    interests = ["politics", "technology", "health", "entertainment", "sports"]
    num_dynamic_users = len(X_train)  # Adjust the number of users
    dynamic_users = create_dynamic_virtual_users(num_dynamic_users, interests)
    social_network = create_social_network(dynamic_users)
    simulation_results = simulate_information_spread(dynamic_users, social_network, pd.read_csv(os.path.join(dataset_path, 'train.csv')))

    # Integrate simulation results into training data
    logging.info("Integrating simulation results into the training data")
    X_train['simulated_influence'] = [len(simulation_results.get(user_id, [])) for user_id in X_train.index]

    # Ensure X_train and y_train have the same number of samples
    if len(X_train) != len(y_train):
        min_len = min(len(X_train), len(y_train))
        X_train = X_train.iloc[:min_len]
        y_train = y_train.iloc[:min_len]

    logging.info("Training model...")
    logging.info(f"X_train (first 5 rows): {X_train.head()}")
    logging.info(f"y_train (first 5 values): {y_train.head()}")

    logging.info(f"X_train shape: {X_train.shape}")
    logging.info(f"y_train shape: {y_train.shape}")

    # Measure model training time
    start_time = time.time()
    try:
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
    except Exception as e:
        logging.error(f"Error during model fitting: {e}")
        logging.error(f"Model type: {type(model)}")
        logging.error(f"X_train type: {type(X_train)}")
        raise

    # Load test data
    X_test, ids = DataLoader.load_test_data(os.path.join(dataset_path, 'test.csv'))
    predictions, prediction_time = Predictor.predict(X_test, ids, model)

    # Load submit data
    submit_data = DataLoader.load_submit_data(os.path.join(dataset_path, 'submit.csv'))
    metrics = Predictor.calculate_metrics(predictions, submit_data)
    metrics['training_time'] = training_time
    metrics['prediction_time'] = prediction_time

    # Analyze simulation impact
    simulation_impact = Predictor.analyze_simulation_impact(predictions, simulation_results, submit_data)
    metrics['simulation_impact'] = simulation_impact

    # Create a unique identifier for the results based on the used parameters
    params_str = '_'.join([f'{k}={v}' for k, v in {**vectorizer_params, **model_params}.items()]) or 'default_params'
    dataset_name = os.path.basename(dataset_path)
    algorithm_dir = os.path.join('results', algorithm_name, dataset_name)
    os.makedirs(algorithm_dir, exist_ok=True)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f'{algorithm_dir}/{algorithm_name}_{params_str}_stats.csv', index=False)
    predictions.to_csv(f'{algorithm_dir}/results_{algorithm_name}_{params_str}.csv', index=False)

    y_true = submit_data['label']
    y_pred = predictions['predicted_label']
    y_proba = predictions['predicted_proba']

    plot_dir = 'plots'
    os.makedirs(plot_dir, exist_ok=True)
    Predictor.plot_confusion_matrix(y_true, y_pred, plot_dir, f'{algorithm_name}_{params_str}_{dataset_name}')
    Predictor.plot_precision_recall_curve(y_true, y_proba, plot_dir, f'{algorithm_name}_{params_str}_{dataset_name}')
    Predictor.plot_roc_curve(y_true, y_proba, plot_dir, f'{algorithm_name}_{params_str}_{dataset_name}')

    logging.info(f"Metrics saved to {algorithm_name}_{params_str}_stats.csv")
    logging.info(f"Results saved to results_{algorithm_name}_{params_str}.csv")


def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

    # Dataset selection
    datasets = ['data/dataset1', 'data/dataset2', 'data/dataset3']
    print("Select dataset:")
    for i, dataset in enumerate(datasets, start=1):
        print(f"{i}. {os.path.basename(dataset)}")
    dataset_choice = int(input("Enter the number of the selected dataset: ")) - 1
    dataset_path = datasets[dataset_choice]

    print("Do you want to test all algorithms? (yes/no)")
    test_all = input().strip().lower() == 'yes'

    if test_all:
        algorithms = [
            'random_forest', 'adaboost', 'gradient_boosting', 'xgboost',
            'logistic_regression', 'voting', 'custom_weighted_voting',
            'pytorch_transformer', 'simple_nn', 'hybrid_ensemble'
        ]
        results = []
        for algorithm in algorithms:
            print(f"Running experiment for {algorithm}")
            run_experiment(algorithm, dataset_path)
            metrics_path = f'results/{algorithm}/{os.path.basename(dataset_path)}/{algorithm}_default_params_stats.csv'
            metrics = pd.read_csv(metrics_path)
            results.append((algorithm, metrics))

        print("Saving comparison of algorithms")
        comparison_df = pd.DataFrame({
            'Algorithm': [res[0] for res in results],
            'Accuracy': [res[1]['accuracy'][0] for res in results],
            'Precision': [res[1]['precision'][0] for res in results],
            'Recall': [res[1]['recall'][0] for res in results],
            'F1 Score': [res[1]['f1_score'][0] for res in results],
            'ROC AUC': [res[1]['roc_auc'][0] for res in results],
            'Training Time': [res[1]['training_time'][0] for res in results],
            'Prediction Time': [res[1]['prediction_time'][0] for res in results]
        })

        comparison_df.to_csv('results/comparison_stats.csv', index=False)

        # Plot comparison chart of algorithm performance
        print("Plotting comparison chart of algorithm performance")
        comparison_df.plot(x='Algorithm', kind='bar', figsize=(10, 8))
        plt.title('Algorithm Performance Comparison')
        plt.ylabel('Scores')
        plt.xticks(rotation=45)
        plt.savefig('results/comparison_plot.png')
        plt.show()

    else:
        print("Select algorithm:")
        print("1. Random Forest")
        print("2. AdaBoost")
        print("3. Gradient Boosting")
        print("4. XGBoost")
        print("5. Logistic Regression")
        print("6. Voting")
        print("7. Custom Weighted Voting")
        print("8. PyTorch Transformer")
        print("9. Simple Neural Network")
        print("10. Hybrid Ensemble Model")
        choice = input("Enter the number of the selected algorithm: ")

        algorithms = {
            '1': 'random_forest',
            '2': 'adaboost',
            '3': 'gradient_boosting',
            '4': 'xgboost',
            '5': 'logistic_regression',
            '6': 'voting',
            '7': 'custom_weighted_voting',
            '8': 'pytorch_transformer',
            '9': 'simple_nn',
            '10': 'hybrid_ensemble'
        }

        algorithm_name = algorithms.get(choice)
        if not algorithm_name:
            print("Invalid choice, select a number from 1 to 10.")
            return

        vectorizer_params = {}
        model_params = {}

        # The user can enter various parameters
        if input("Do you want to change the max_features value? (yes/no): ").strip().lower() == 'yes':
            vectorizer_params['max_features'] = int(input("Enter the max_features value: "))

        if choice in ['1', '2', '3', '4', '7']:  # Algorithms with n_estimators parameter
            if input("Do you want to change the n_estimators value? (yes/no): ").strip().lower() == 'yes':
                model_params['n_estimators'] = int(input("Enter the n_estimators value: "))

        print(f"Running experiment for {algorithm_name}")
        run_experiment(algorithm_name, dataset_path, vectorizer_params, model_params)

    # Social network simulation integration
    interests = ["politics", "technology", "health", "entertainment", "sports"]
    num_dynamic_users = 10  # Adjust the number of dynamic users
    print(f"Creating {num_dynamic_users} dynamic users for social network simulation")
    dynamic_users = create_dynamic_virtual_users(num_dynamic_users, interests)
    social_network = create_social_network(dynamic_users)

    # Example of information spread simulation
    print("Simulating information spread")
    df = pd.read_csv(os.path.join(dataset_path, 'train.csv'))  # Dummy dataframe to simulate articles
    simulation_results = simulate_information_spread(dynamic_users, social_network, df)

    # Visualization of simulation results
    print("Plotting simulation results")
    results_dir = 'results'
    os.makedirs(results_dir, exist_ok=True)
    plot_simulation_results(simulation_results, num_dynamic_users, results_dir)
    plot_article_read_share_frequency(simulation_results, results_dir)
    plot_daily_infections(simulation_results, results_dir)
    plot_cumulative_infections(simulation_results, results_dir)


if __name__ == "__main__":
    main()
