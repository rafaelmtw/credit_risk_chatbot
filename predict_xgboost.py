
import joblib
import numpy as np

# Define a function to load the model and predict new data
def predict_new_data(new_data, model_file="xgboost_model.pkl"):
    """Load the saved model and predict on new data."""
    # Load the model
    model = joblib.load(model_file)

    # Ensure new data is a 2D array
    if not isinstance(new_data, np.ndarray):
        new_data = np.array(new_data)
    if new_data.ndim == 1:
        new_data = new_data.reshape(1, -1)

    # Predict and return the results
    predictions = model.predict(new_data)
    probabilities = model.predict_proba(new_data)
    return predictions, probabilities

# Example usage
if __name__ == "__main__":
    # Example new input data for prediction
    new_input = [67,2,7.063903961472068,6,False,False,False,False,True,False,False,True,True,False,False,True,False,False,False,False,False,False,False,True]
    predictions, probabilities = predict_new_data(new_input)

    print("Predictions:", predictions)
    print("Probabilities:", probabilities)
