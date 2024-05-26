import numpy as np
import pandas as pd
from collections import Counter

# L - Load the data from CSV


def load_data(file_path):
    return pd.read_csv(file_path)

# I - Initialize the value of k


def initialize_k():
    return 3  # Choose the number of neighbors

# S - Sort distances in ascending order


def calculate_distances(train_data, test_point):
    distances = []
    for i in range(len(train_data)):
        distance = np.sqrt(
            np.sum((np.array(test_point[:-1]) - np.array(train_data.iloc[i, :-1])) ** 2))
        distances.append((distance, train_data.iloc[i, -1]))
    distances.sort(key=lambda x: x[0])
    return distances

# K - Keep the top k nearest neighbors


def get_top_k_neighbors(distances, k):
    return distances[:k]

# S - Select the most frequent class


def select_most_frequent_class(neighbors):
    neighbor_classes = [neighbor[1] for neighbor in neighbors]
    most_common = Counter(neighbor_classes).most_common(1)
    return most_common[0][0]

# M - Make predictions based on neighbors


def knn_predict(train_data, test_point, k):
    distances = calculate_distances(train_data, test_point)
    neighbors = get_top_k_neighbors(distances, k)
    return select_most_frequent_class(neighbors)

# I - Iterate through each data point for classification


def classify_data(train_data, test_points, k):
    predictions = []
    for test_point in test_points:
        predicted_class = knn_predict(train_data, test_point, k)
        predictions.append(predicted_class)
    return predictions

# R - Return the predicted class


def main():
    file_path = 'iris.csv'  # Path to your CSV file
    data = load_data(file_path)

    # Assuming the last column is the class label
    train_data = data.iloc[:, :-1]
    # Using the same data for simplicity, normally this should be separate
    test_data = data.iloc[:, :-1]

    # Add an empty column for class in test data
    test_data['Class'] = np.nan

    # Initialize k
    k = initialize_k()

    # Classify the test data
    predictions = classify_data(data, test_data.values, k)

    # Print the predictions
    for test_point, prediction in zip(test_data.values, predictions):
        print(
            f"The predicted class for the test point {test_point[:-1]} is: {prediction}")


if __name__ == "__main__":
    main()
