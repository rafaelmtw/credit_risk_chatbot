
import joblib
import numpy as np
import pandas as pd

# Define a function to load the model and predict new data
def predict_new_data(new_data, model_file="gnb_model.pkl"):
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
    # Example usage of the prediction function
    # Replace `new_input` with your own data point(s)
    feature_names = ["Age","Job","Credit amount","Duration","Purpose_car","Purpose_domestic appliances","Purpose_education","Purpose_furniture/equipment","Purpose_radio/TV","Purpose_repairs","Purpose_vacation/others","Sex_male","Housing_own","Housing_rent","Savings_moderate","Savings_no_inf","Savings_quite rich","Savings_rich","Check_moderate","Check_no_inf","Check_rich","Age_cat_Young","Age_cat_Adult","Age_cat_Senior"]
    new_input = [67,2,7.063903961472068,6,False,False,False,False,True,False,False,True,True,False,False,True,False,False,False,False,False,False,False,True]

    new_input_df = pd.DataFrame([new_input], columns=feature_names)
    predictions, probabilities = predict_new_data(new_input_df)

    print("Predictions:", predictions)
    print("Probabilities:", probabilities)
