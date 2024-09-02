# Pima-Diabetes-Prediction
### Diabetes Prediction Using Artificial Neural Network (ANN)

### **Objective**
The objective of this project is to build an Artificial Neural Network (ANN) model to predict the likelihood of diabetes in patients based on certain medical features. The model is trained using the Pima Indians Diabetes dataset and provides predictions on whether a person has diabetes or not.

### **Introduction to Data Columns**
The dataset used in this project contains several medical diagnostic features, which are:

1. Pregnancies: Number of times the patient has been pregnant.
2. Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test.
3. BloodPressure: Diastolic blood pressure (mm Hg).
4. SkinThickness: Triceps skinfold thickness (mm).
5. Insulin: 2-Hour serum insulin (mu U/ml).
6. BMI: Body mass index (weight in kg/(height in m)^2).
7. DiabetesPedigreeFunction: A function that scores likelihood of diabetes based on family history.
8. Age: Age of the patient (years).
9. Outcome: Class variable (0 or 1) where 1 indicates that the patient has diabetes and 0 indicates they do not.

### **Model Output**
The model outputs a prediction for the Outcome variable, which indicates whether the patient is likely to have diabetes:

0: The patient does not have diabetes.
1: The patient has diabetes.

## How to Use This Repository

1. Clone the Repository
```bash
git clone https://github.com/yourusername/diabetes-prediction-ann.git
cd diabetes-prediction-ann

### Make a Prediction for New Data

To make a prediction for a new data point using the trained model, follow these steps:

1. **Load the Model**: Load the trained model from the saved file.
2. **Prepare the New Data**: Create a tensor from the new data point you want to predict.
3. **Make the Prediction**: Pass the new data through the model to get the prediction.

Here’s the code to do it:

```python
import torch

# Load the trained model
model = torch.load('diabetes.pt')
model.eval()  # Set the model to evaluation mode

# New data point (example values for each feature)
new_data = [6.0, 130.0, 72.0, 40.0, 0.0, 25.6, 0.627, 45.0]

# Convert the new data to a tensor
new_data_tensor = torch.tensor(new_data)

# Make a prediction
with torch.no_grad():  # Disable gradient calculation for inference
    prediction = model(new_data_tensor).argmax().item()

# Print the result
if prediction == 1:
    print("The model predicts: Diabetic")
else:
    print("The model predicts: Non-Diabetic")
