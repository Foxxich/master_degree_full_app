import os
import shutil
import time
from matplotlib import pyplot as plt
import matplotlib
matplotlib.use('Agg') 
import pandas as pd
from django.shortcuts import render
from django.conf import settings
from .forms import UploadFileForm
import importlib
import logging

from .ml.data_loader import DataLoader
from .ml.predict import Predictor

from channels.layers import get_channel_layer
from asgiref.sync import async_to_sync

import sys

# Add the path for the 'mlapp/ml' module
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'mlapp', 'ml'))

# Import the necessary functions from your social_network.py
from .ml.social_network import create_dynamic_virtual_users, create_social_network, simulate_information_spread, \
    plot_simulation_results, plot_article_read_share_frequency, plot_daily_infections, plot_cumulative_infections

# Initialize logging
logger = logging.getLogger(__name__)

def clear_media_folder():
    media_path = settings.MEDIA_ROOT
    if os.path.exists(media_path):
        shutil.rmtree(media_path)  # Remove the entire media folder
    os.makedirs(media_path)  # Recreate the media folder

def upload_file(request):
    if request.method == 'POST':
        clear_media_folder()  # Clear the media folder
        channel_layer = get_channel_layer()

        if not channel_layer:
            logger.error("Channel layer is not available.")
            raise RuntimeError("Channel layer is not available. Please check your Channels configuration.")

        def log_message(message):
            # Send message to WebSocket group
            async_to_sync(channel_layer.group_send)(
                'logs',
                {
                    'type': 'send_log',
                    'message': message
                }
            )
            logger.info(message)

        dataset = request.POST['dataset']
        algorithm = request.POST['algorithm']

        # The datasets are already defined, and their paths are known
        absolute_dataset_path = os.path.normpath(os.path.join(settings.BASE_DIR, 'mlapp', 'ml', 'data', dataset))
        train_path = os.path.join(absolute_dataset_path, 'train.csv')
        test_path = os.path.join(absolute_dataset_path, 'test.csv')
        submit_path = os.path.join(absolute_dataset_path, 'submit.csv')

        algorithms = [
            'random_forest', 'adaboost', 'gradient_boosting', 'xgboost',
            'logistic_regression', 'voting', 'custom_weighted_voting',
            'pytorch_transformer', 'simple_nn', 'hybrid_ensemble'
        ]

        plot_paths = []

        if algorithm == 'all':
            results = []
            for algo in algorithms:
                log_message(f"Starting fake news detection process with {algo} on {dataset}...")

                vectorizer_params = {}
                model_params = {}

                module = importlib.import_module(f'mlapp.ml.algorithms.{algo}')
                model = module.get_model(**vectorizer_params, **model_params)

                X_train, y_train = DataLoader.load_train_data(train_path)
                start_time = time.time()
                model.fit(X_train, y_train)
                training_time = time.time() - start_time
                log_message(f"Model training for {algo} completed in {training_time:.4f} seconds")

                X_test, ids = DataLoader.load_test_data(test_path)
                predictions, prediction_time = Predictor.predict(X_test, ids, model)
                log_message(f"Predictions for {algo} completed in {prediction_time:.4f} seconds")

                submit_data = DataLoader.load_submit_data(submit_path)
                metrics = Predictor.calculate_metrics(predictions, submit_data)
                metrics['training_time'] = training_time
                metrics['prediction_time'] = prediction_time

                params_str = '_'.join([f'{k}={v}' for k, v in {**vectorizer_params, **model_params}.items()]) or 'default_params'
                algorithm_dir = os.path.join(settings.MEDIA_ROOT, 'results', algo, dataset)
                os.makedirs(algorithm_dir, exist_ok=True)

                metrics_df = pd.DataFrame([metrics])
                metrics_path = os.path.join(algorithm_dir, f'{algo}_{params_str}_stats.csv')
                metrics_df.to_csv(metrics_path, index=False)

                results_path = os.path.join(algorithm_dir, f'results_{algo}_{params_str}.csv')
                predictions.to_csv(results_path, index=False)

                plot_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
                os.makedirs(plot_dir, exist_ok=True)
                y_true = submit_data['label']
                y_pred = predictions['predicted_label']
                y_proba = predictions['predicted_proba']

                confusion_matrix_path = Predictor.plot_confusion_matrix(y_true, y_pred, plot_dir, f'{algo}_{params_str}_{dataset}')
                precision_recall_curve_path = Predictor.plot_precision_recall_curve(y_true, y_proba, plot_dir, f'{algo}_{params_str}_{dataset}')
                roc_curve_path = Predictor.plot_roc_curve(y_true, y_proba, plot_dir, f'{algo}_{params_str}_{dataset}')

                plot_paths.extend(filter(None, [confusion_matrix_path, precision_recall_curve_path, roc_curve_path]))

                metrics_df = pd.read_csv(metrics_path)
                results.append((algo, metrics_df))

            # Plot the comparison of algorithms
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

            comparison_df.plot(x='Algorithm', kind='bar', figsize=(10, 8))
            plt.title('Algorithm Performance Comparison')
            plt.ylabel('Scores')
            plt.xticks(rotation=45)
            comparison_plot_path = os.path.join(plot_dir, 'comparison_plot.png')
            plt.savefig(comparison_plot_path)
            plt.close()
            plot_paths.append(comparison_plot_path)

            log_message("All algorithms have finished running.")
        else:
            log_message(f"Starting fake news detection process with {algorithm} on {dataset}...")

            vectorizer_params = {}
            model_params = {}

            module = importlib.import_module(f'mlapp.ml.algorithms.{algorithm}')
            model = module.get_model(**vectorizer_params, **model_params)

            X_train, y_train = DataLoader.load_train_data(train_path)
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time
            log_message(f"Model training completed in {training_time:.4f} seconds")

            X_test, ids = DataLoader.load_test_data(test_path)
            predictions, prediction_time = Predictor.predict(X_test, ids, model)
            log_message(f"Predictions completed in {prediction_time:.4f} seconds")

            submit_data = DataLoader.load_submit_data(submit_path)
            metrics = Predictor.calculate_metrics(predictions, submit_data)
            metrics['training_time'] = training_time
            metrics['prediction_time'] = prediction_time

            params_str = '_'.join([f'{k}={v}' for k, v in {**vectorizer_params, **model_params}.items()]) or 'default_params'
            algorithm_dir = os.path.join(settings.MEDIA_ROOT, 'results', algorithm, dataset)
            os.makedirs(algorithm_dir, exist_ok=True)

            metrics_df = pd.DataFrame([metrics])
            metrics_path = os.path.join(algorithm_dir, f'{algorithm}_{params_str}_stats.csv')
            metrics_df.to_csv(metrics_path, index=False)

            results_path = os.path.join(algorithm_dir, f'results_{algorithm}_{params_str}.csv')
            predictions.to_csv(results_path, index=False)

            plot_dir = os.path.join(settings.MEDIA_ROOT, 'plots')
            os.makedirs(plot_dir, exist_ok=True)
            y_true = submit_data['label']
            y_pred = predictions['predicted_label']
            y_proba = predictions['predicted_proba']

            confusion_matrix_path = Predictor.plot_confusion_matrix(y_true, y_pred, plot_dir, f'{algorithm}_{params_str}_{dataset}')
            precision_recall_curve_path = Predictor.plot_precision_recall_curve(y_true, y_proba, plot_dir, f'{algorithm}_{params_str}_{dataset}')
            roc_curve_path = Predictor.plot_roc_curve(y_true, y_proba, plot_dir, f'{algorithm}_{params_str}_{dataset}')

            plot_paths.extend(filter(None, [confusion_matrix_path, precision_recall_curve_path, roc_curve_path]))

            log_message("Metrics and plots generated successfully.")

        # Social network simulation integration
        log_message("Starting social network simulation...")
        interests = ["politics", "technology", "health", "entertainment", "sports"]
        num_dynamic_users = len(X_train)  # Adjust the number of users
        dynamic_users = create_dynamic_virtual_users(num_dynamic_users, interests)
        social_network = create_social_network(dynamic_users)
        simulation_results = simulate_information_spread(dynamic_users, social_network, pd.read_csv(train_path))

        # Plot and save simulation results
        plot_simulation_results(simulation_results, num_dynamic_users, plot_dir)
        plot_article_read_share_frequency(simulation_results, plot_dir)
        plot_daily_infections(simulation_results, plot_dir)
        plot_cumulative_infections(simulation_results, plot_dir)

        # Fetch all images from media/plots directory for the results page
        images = [f for f in os.listdir(plot_dir) if os.path.isfile(os.path.join(plot_dir, f))]

        return render(request, 'results.html', {
            'message': 'Algorithm has been processed successfully.' if algorithm != 'all' else 'All algorithms have been processed successfully.',
            'metrics_path': metrics_path if algorithm != 'all' else None,
            'results_path': results_path if algorithm != 'all' else None,
            'plot_paths': [os.path.join(settings.MEDIA_URL, 'plots', os.path.basename(path)) for path in plot_paths],
            'images': images,
            'media_url': settings.MEDIA_URL + 'plots/',  # Construct the URL path for images
        })
    else:
        form = UploadFileForm()
    return render(request, 'upload.html', {'form': form})
