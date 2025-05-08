import numpy as np
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import time
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import os

from helper import extract_feature, save_model
from preprocess import get_training_data


def save_training_plot(model_name, train_time, predict_time, accuracy, save_time):
    # Create results directory if it doesn't exist
    if not os.path.exists('results'):
        os.makedirs('results')
    
    # Time data for plotting
    times = [train_time, predict_time, save_time]
    labels = ['Training', 'Prediction', 'Saving']
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Time distribution plot
    plt.subplot(1, 2, 1)
    plt.bar(labels, times)
    plt.title(f'{model_name} Time Distribution')
    plt.ylabel('Time (seconds)')
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.bar(['Accuracy'], [accuracy])
    plt.title(f'{model_name} Accuracy')
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(f'results/{model_name.lower()}_training_results.png')
    plt.close()


def _get_training_feature():
    print("Loading training data...")
    start_time = time.time()
    training_images, training_labels = get_training_data()
    load_time = time.time() - start_time
    print(f"Number of training images: {len(training_images)}")
    print(f"Data loading time: {load_time:.2f} seconds")

    print("Extracting features...")
    start_time = time.time()
    img_features, codebook = extract_feature(training_images)
    feature_time = time.time() - start_time
    print(f"Extracted feature dimensions: {img_features.shape}")
    print(f"Feature extraction time: {feature_time:.2f} seconds")
    
    return img_features, codebook, training_labels


def train_svm(model_directory: str = "models"):
    print("\n=== Starting SVM Model Training ===")
    total_start_time = time.time()
    
    img_features, codebook, training_labels = _get_training_feature()

    # Train SVM classifier
    print("Training SVM model...")
    start_time = time.time()
    model = SVC(max_iter=10000)
    estimator = model.fit(img_features, training_labels)
    train_time = time.time() - start_time
    
    # Calculate training accuracy
    print("Calculating accuracy...")
    start_time = time.time()
    predictions = estimator.predict(img_features)
    accuracy = accuracy_score(training_labels, predictions)
    predict_time = time.time() - start_time
    
    total_time = time.time() - total_start_time
    print(f"Training completed!")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Prediction time: {predict_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Training accuracy: {accuracy:.4f}")

    # Save model for testing and reuse
    print("Saving model...")
    start_time = time.time()
    save_model(model_directory + "/svm_estimator.pkl", estimator)
    save_model(model_directory + "/svm_codebook.pkl", codebook)
    save_time = time.time() - start_time
    print(f"Model saved successfully! (Time: {save_time:.2f} seconds)")
    
    # Save training results plot
    save_training_plot('SVM', train_time, predict_time, accuracy, save_time)


def train_nb(model_directory: str = "models"):
    print("\n=== Starting Naive Bayes Model Training ===")
    total_start_time = time.time()
    
    img_features, codebook, training_labels = _get_training_feature()

    # Train Naive Bayes classifier
    print("Training Naive Bayes model...")
    start_time = time.time()
    model = GaussianNB()
    estimator = model.fit(img_features, training_labels)
    train_time = time.time() - start_time
    
    # Calculate training accuracy
    print("Calculating accuracy...")
    start_time = time.time()
    predictions = estimator.predict(img_features)
    accuracy = accuracy_score(training_labels, predictions)
    predict_time = time.time() - start_time
    
    total_time = time.time() - total_start_time
    print(f"Training completed!")
    print(f"Training time: {train_time:.2f} seconds")
    print(f"Prediction time: {predict_time:.2f} seconds")
    print(f"Total time: {total_time:.2f} seconds")
    print(f"Training accuracy: {accuracy:.4f}")

    # Save model for testing and reuse
    print("Saving model...")
    start_time = time.time()
    save_model(model_directory + "/nb_estimator.pkl", estimator)
    save_model(model_directory + "/nb_codebook.pkl", codebook)
    save_time = time.time() - start_time
    print(f"Model saved successfully! (Time: {save_time:.2f} seconds)")
    
    # Save training results plot
    save_training_plot('Naive Bayes', train_time, predict_time, accuracy, save_time)


if __name__ == "__main__":
    train_svm()
    train_nb()
