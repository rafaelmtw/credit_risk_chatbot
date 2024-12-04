import joblib
import numpy as np
import pandas as pd

def remove_punctuation(s):
    if type(s) == str: 
        translator = str.maketrans('', '', string.punctuation)
        s = s.translate(translator)
    return s

def HKD_to_DM(hkd):
    dm = hkd*0.234
    return dm

def conversion(result):
    result['SAVING ACCOUNT BALANCE'] = HKD_to_DM(result['SAVING ACCOUNT BALANCE'])
    result['CHECKING ACCOUNT BALANCE'] = HKD_to_DM(result['CHECKING ACCOUNT BALANCE'])
    result['CREDIT AMOUNT'] = HKD_to_DM(result['CREDIT AMOUNT'])
    return result

# Define a function to load the model and predict new data
def predict_new_data(new_data, model_file="./model/gnb_model.pkl"):
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

# Main function
def predict_result(result):

    result['SAVING ACCOUNT BALANCE'] = float(remove_punctuation(result['SAVING ACCOUNT BALANCE']))
    result['CHECKING ACCOUNT BALANCE'] = float(remove_punctuation(result['CHECKING ACCOUNT BALANCE']))
    result['CREDIT AMOUNT'] = float(remove_punctuation(result['CREDIT AMOUNT']))

    age = float(result['AGE'])
    job = result['JOB and RESIDENCY']
    credit_amount = np.log(result['CREDIT AMOUNT'])
    duration = float(result['CREDIT DURATION'])
    # Purpose: ['radio/TV' 'education' 'furniture/equipment' 'car' 'business' 'domestic appliances' 'repairs' 'vacation/others']
    Purpose_car = (1 if result['PURPOSE OF CREDIT'].lower() == 'car' else 0)
    Purpose_domestic_appliances = (1 if result['PURPOSE OF CREDIT'].lower() == 'domestic appliances' else 0)
    Purpose_education = (1 if result['PURPOSE OF CREDIT'].lower() == 'education' else 0)
    Purpose_furniture_equipment = (1 if result['PURPOSE OF CREDIT'].lower() == 'furniture/equipment' else 0)
    Purpose_radio_TV = (1 if result['PURPOSE OF CREDIT'].lower() == 'radio/TV' else 0) 
    Purpose_repairs = (1 if result['PURPOSE OF CREDIT'].lower() == 'repairs' else 0)
    Purpose_vacation_others = (1 if result['PURPOSE OF CREDIT'].lower() == 'vacation/others' else 0)
    # Sex
    Sex_male = (1 if float(result['SEX']) == '1' else 0)
    # Housing
    Housing_own = (1 if result['HOUSING'].lower() == 'own' else 0)
    Housing_rent = (1 if result['HOUSING'].lower() == 'rent' else 0)
    # Saving Account Balance
    Savings_moderate = (1 if (result['SAVING ACCOUNT BALANCE']  >= 100 and result['SAVING ACCOUNT BALANCE'] < 500) else 0)
    Savings_no_inf = 0
    Savings_quite_rich = (1 if (result['SAVING ACCOUNT BALANCE'] >= 500 and result['SAVING ACCOUNT BALANCE'] < 1000) else 0)
    Savings_rich = (1 if result['SAVING ACCOUNT BALANCE'] >= 1000 else 0)
    # Checking Account
    Check_moderate = (1 if (result['CHECKING ACCOUNT BALANCE'] >= 0 and result['CHECKING ACCOUNT BALANCE'] < 200) else 0)
    Check_no_inf = 0
    Check_rich = (1 if (result['CHECKING ACCOUNT BALANCE'] >= 200) else 0)
    # Age
    Age_cat_Young = (1 if (age >= 25 and age < 35) else 0)
    Age_cat_Adult = (1 if (age >= 35 and age < 60) else 0)
    Age_cat_Senior = (1 if (age >= 60) else 0)

    new_input = [age, job, credit_amount, duration, Purpose_car, Purpose_domestic_appliances, Purpose_education, Purpose_furniture_equipment, Purpose_radio_TV, Purpose_repairs, Purpose_vacation_others, Sex_male, Housing_own, Housing_rent, Savings_moderate, Savings_no_inf, Savings_quite_rich, Savings_rich, Check_moderate, Check_no_inf, Check_rich, Age_cat_Young, Age_cat_Adult, Age_cat_Senior]
    # Replace `new_input` with your own data point(s)
    feature_names = ["Age","Job","Credit amount","Duration","Purpose_car","Purpose_domestic appliances","Purpose_education","Purpose_furniture/equipment","Purpose_radio/TV","Purpose_repairs","Purpose_vacation/others","Sex_male","Housing_own","Housing_rent","Savings_moderate","Savings_no_inf","Savings_quite rich","Savings_rich","Check_moderate","Check_no_inf","Check_rich","Age_cat_Young","Age_cat_Adult","Age_cat_Senior"]
    # new_input = [67,2,7.063903961472068,6,False,False,False,False,True,False,False,True,True,False,False,True,False,False,False,False,False,False,False,True]

    new_input_df = pd.DataFrame([new_input], columns=feature_names)
    predictions, probabilities = predict_new_data(new_input_df)

    print("Predictions:", predictions)
    print("Probabilities:", probabilities)
