from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
import pickle
import os


def get_classification_data(test_size=0.2):
    # 1. Load Data
    iris = load_iris()
    X = iris.data
    y = iris.target

    # 2. Split Data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    # 3. Feature Scaling (Important for MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, y_train, X_test_scaled, y_test


if __name__ == "__main__":
    
    X_train_scaled, y_train, X_test_scaled, y_test = get_classification_data()

    # 4. Choose and Train the Model (MLP)
    model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=42) # Adjust parameters
    model.fit(X_train_scaled, y_train)

    # 5. Make Predictions
    y_pred = model.predict(X_test_scaled)

    # 6. Evaluate the Model
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    #Example of predicting probabilities
    probabilities = model.predict_proba(X_test_scaled)
    print(f"Probabilities for test set first 5 rows:\n {probabilities[:5]}")

    model_name = "iris_classifcation.pkl"
    pth = os.path.join(os.getcwd(), 'model_dump', model_name)

    pickle.dump(model, open(pth,'wb'))
    print(f"{model_name} save {pth} successfully")