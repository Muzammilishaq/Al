import numpy as np
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Step 1: Create a sample dataset
data = pd.DataFrame({
    'Feature1': [5, 2, 8, 7, 3, 4, 6, 7, 8, 5],
    'Feature2': [1, 9, 2, 4, 6, 7, 8, 2, 3, 5],
    'Class': ['A', 'B', 'A', 'A', 'B', 'B', 'A', 'A', 'B', 'B']
})

# Step 2: Separate features and target label
X = data[['Feature1', 'Feature2']].values
y = data['Class'].values

# Step 3: Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Define the Euclidean distance function
def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

# Step 5: Implement the KNN algorithm
def knn_predict(X_train, y_train, x_query, k=3):
    distances = []
    
    # Calculate the distance between x_query and all points in X_train
    for i in range(len(X_train)):
        dist = euclidean_distance(X_train[i], x_query)
        distances.append((dist, y_train[i]))
    
    # Sort distances and get the nearest k neighbors
    distances = sorted(distances)[:k]
    nearest_neighbors = [label for _, label in distances]
    
    # Return the most common class label among the nearest neighbors
    prediction = Counter(nearest_neighbors).most_common(1)[0][0]
    return prediction

# Step 6: Make predictions on the test set
k = 3  # Set the number of neighbors
y_pred = [knn_predict(X_train, y_train, x_test, k) for x_test in X_test]

# Step 7: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy * 100:.2f}%")
print("Confusion Matrix:")
print(conf_matrix)

# Step 8: Test with a new sample (optional)
# Example query point, replace with your own values
query_point = np.array([3, 7])
predicted_class = knn_predict(X_train, y_train, query_point, k)
print(f"Predicted class for the query point {query_point} is: {predicted_class}")
