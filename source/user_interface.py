import tkinter as tk
from tkinter import messagebox, ttk

import pandas as pd
import torch
import torch.nn as nn
from joblib import load


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# read the values from scaler_means_and_stds.csv
import csv

with open('source/scaler_means_stds.csv') as csv_file:
    reader = csv.reader(csv_file)
    scaler_dict = dict(reader)

age_mean = float(scaler_dict['age_mean'])
age_std = float(scaler_dict['age_std'])
glucose_mean = float(scaler_dict['glucose_mean'])
glucose_std = float(scaler_dict['glucose_std'])
bmi_mean = float(scaler_dict['bmi_mean'])
bmi_std = float(scaler_dict['bmi_std'])

# Function to handle the 'Predict' button click
def on_predict():
    # Collect the input data from the entry fields
    input_data = {label: entry.get() for label, entry in entries.items()}

    # Validate Age
    try:
        age = float(input_data['Age'])
        if age < 0:
            messagebox.showerror("Invalid Input", "Age cannot be negative.")
            return
    except ValueError:
        messagebox.showerror("Invalid Input", "Age must be a number.")
        return

    # Validate BMI
    try:
        bmi = float(input_data['BMI'])
        if not (0 <= bmi <= 100):
            messagebox.showerror("Invalid Input", "BMI must be between 0 and 100.")
            return
    except ValueError:
        messagebox.showerror("Invalid Input", "BMI must be a number.")
        return

    # Validate Average Glucose Level
    try:
        avg_glucose_level = float(input_data['Avg Glucose Level'])
        if not (40 <= avg_glucose_level <= 280):
            messagebox.showerror("Invalid Input", "Average Glucose Level must be between 40 and 280.")
            return
    except ValueError:
        messagebox.showerror("Invalid Input", "Average Glucose Level must be a number.")
        return

    #error if an input is None
    for label, entry in entries.items():
        if entry.get() == '':
            messagebox.showerror("Invalid Input", f"{label} cannot be empty.")
            return

    # Preprocess the data
    # Encode categorical as numbers
    input_data['Gender'] = 0 if input_data['Gender'] == 'Female' else 1  # Assuming no 'Other' option in UI
    input_data['Ever Married'] = 1 if input_data['Ever Married'] == 'Yes' else 0
    input_data['Hypertension'] = 1 if input_data['Hypertension'] == 'Yes' else 0
    input_data['Previous Heart Disease'] = 1 if input_data['Previous Heart Disease'] == 'Yes' else 0
    input_data['Residence Type'] = 1 if input_data['Residence Type'] == 'Urban' else 0
    input_data['Smoking Status'] = {'formerly smoked': 0, 'never smoked': 1, 'smokes': 2, 'Unknown': 3}[input_data['Smoking Status']]
    input_data['Work Type'] = {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'Children': 3, 'Never_worked': 4}[input_data['Work Type']]

    # One hot encoding
    # One-Hot Encoding for work_type (assuming a dropdown for work_type in your UI)
    work_type_categories = [1, 2, 3, 4]  # 'Private' is the dropped category
    work_type_input = input_data['Work Type']
    for category in work_type_categories:
        input_data[f'work_type_{str(category)}'] = 1 if work_type_input == category else 0

    # One-Hot Encoding for smoking_status
    smoking_status_categories = [1, 2, 3]  # 'formerly smoked' is the dropped category
    smoking_status_input = input_data['Smoking Status']
    for category in smoking_status_categories:
        input_data[f'smoking_status_{str(category)}'] = 1 if smoking_status_input == category else 0

    gender_categories = [0, 1]
    gender_input = input_data['Gender']
    for category in gender_categories:
        input_data[f'gender_{str(category)}'] = 1 if gender_input == category else 0


    # Remove original categorical variables
    del input_data['Work Type']
    del input_data['Smoking Status']
    del input_data['Gender']

    # Scaler
    input_data['Age'] = (float(input_data['Age']) - age_mean) / age_std
    input_data['Avg Glucose Level'] = (float(input_data['Avg Glucose Level']) - glucose_mean) / glucose_std
    input_data['BMI'] = (float(input_data['BMI']) - bmi_mean) / bmi_std

    # print(input_data)

    # Load your models (Random Forest and Neural Network)
    random_forest_model = load('models/random_forest_model.joblib')
    neural_network_model = SimpleNN(16, 32, 1)
    neural_network_model.load_state_dict(torch.load('models/neural_network_model.pth'))
    neural_network_model.eval()  # Set the model to evaluation mode

    # Assuming 'input_data' contains the input values

    # Define the input values and column names
    input_values = [
        float(input_data['Age']),
        float(input_data['Hypertension']),
        float(input_data['Previous Heart Disease']),
        float(input_data['Ever Married']),
        float(input_data['Residence Type']),
        float(input_data['Avg Glucose Level']),
        float(input_data['BMI']),
        float(input_data['work_type_1']),
        float(input_data['work_type_2']),
        float(input_data['work_type_3']),
        float(input_data['work_type_4']),
        float(input_data['smoking_status_1']),
        float(input_data['smoking_status_2']),
        float(input_data['smoking_status_3']),
        float(input_data['gender_0']),
        float(input_data['gender_1'])
    ]

    column_names = [
        'age', 'hypertension', 'heart_disease', 'ever_married', 'Residence_type',
        'avg_glucose_level', 'bmi', 'work_type_1', 'work_type_2', 'work_type_3', 'work_type_4',
        'smoking_status_1', 'smoking_status_2', 'smoking_status_3', 'gender_0', 'gender_1'
    ]

    # Create a DataFrame with the specified columns and input values
    df_rf = pd.DataFrame([input_values], columns=column_names)


    # Make predictions with the models
    random_forest_prediction = random_forest_model.predict(df_rf)[0]
    stroke_bool = True if random_forest_prediction == 1 else False

    # Prepare input data for the Neural Network model
    input_tensor = torch.tensor([
            float(input_data['Age']),
            float(input_data['Hypertension']),
            float(input_data['Previous Heart Disease']),
            float(input_data['Ever Married']),
            float(input_data['Residence Type']),
            float(input_data['Avg Glucose Level']),
            float(input_data['BMI']),
            float(input_data['work_type_1']),
            float(input_data['work_type_2']),
            float(input_data['work_type_3']),
            float(input_data['work_type_4']),
            float(input_data['smoking_status_1']),
            float(input_data['smoking_status_2']),
            float(input_data['smoking_status_3']),
            float(input_data['gender_0']),
            float(input_data['gender_1'])
            ])

    # Forward pass to get the prediction from the Neural Network model
    neural_network_model.eval()  # Set to evaluation mode
    with torch.no_grad():
        neural_network_prediction = neural_network_model(input_tensor)

    # Convert the prediction to a float
    neural_network_prediction = neural_network_prediction.item()
    # TODO: Display the prediction in a messagebox
    if stroke_bool:
        messagebox.showinfo("Stroke Likelihood Prediction", f"RF predicts stroke. According to neural network there is {neural_network_prediction * 100:.2f}% likelihood of stroke.")
    else:
        messagebox.showinfo("Stroke Likelihood Prediction", f"RF predicts no stroke. According to neural network there is {neural_network_prediction * 100:.2f}% likelihood of stroke.")


    pass


# Creating the main window
root = tk.Tk()
root.title("Stroke Prediction System")

# Adding a frame to hold the input fields and buttons
frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

# Adding input fields for each feature
labels = ['Age', 'BMI', 'Hypertension', 'Previous Heart Disease', 'Avg Glucose Level', 'Gender', 'Ever Married', 'Work Type', 'Residence Type', 'Smoking Status']
entries = {}

# TODO: should there be other?
gender_options = ["Female", "Male"]
heart_disease_options = ["Yes", "No"]
residence_type_options = ["Urban", "Rural"]
smoking_status_options = ['formerly smoked', 'never smoked', 'smokes', 'Unknown']
ever_married_options = ["Yes", "No"]
work_type_options = ["Private", "Self-employed", "Govt_job", "Children", "Never_worked"]
hypertension_options = ["Yes", "No"]

# Add the input fields to the frame
for i, label in enumerate(labels):
    ttk.Label(frame, text=label).grid(row=i, column=0, sticky=tk.W, padx=5, pady=5)
    if label == 'Gender':
        gender_combobox = ttk.Combobox(frame, values=gender_options, state="readonly")
        gender_combobox.grid(row=i, column=1, sticky=tk.E, padx=5, pady=5)
        gender_combobox.set(gender_options[0])  # Set default value
        entries[label] = gender_combobox
    elif label == 'Previous Heart Disease':
        heart_disease_combobox = ttk.Combobox(frame, values=heart_disease_options, state="readonly")
        heart_disease_combobox.grid(row=i, column=1, sticky=tk.E, padx=5, pady=5)
        heart_disease_combobox.set(heart_disease_options[0])
        entries[label] = heart_disease_combobox
    elif label == 'Residence Type':
        residence_type_combobox = ttk.Combobox(frame, values=residence_type_options, state="readonly")
        residence_type_combobox.grid(row=i, column=1, sticky=tk.E, padx=5, pady=5)
        residence_type_combobox.set(residence_type_options[0])
        entries[label] = residence_type_combobox
    elif label == 'Smoking Status':
        smoking_status_combobox = ttk.Combobox(frame, values=smoking_status_options, state="readonly")
        smoking_status_combobox.grid(row=i, column=1, sticky=tk.E, padx=5, pady=5)
        smoking_status_combobox.set(smoking_status_options[0])
        entries[label] = smoking_status_combobox
    elif label == 'Ever Married':
        ever_married_combobox = ttk.Combobox(frame, values=ever_married_options, state="readonly")
        ever_married_combobox.grid(row=i, column=1, sticky=tk.E, padx=5, pady=5)
        ever_married_combobox.set(ever_married_options[0])
        entries[label] = ever_married_combobox
    elif label == 'Work Type':
        work_type_combobox = ttk.Combobox(frame, values=work_type_options, state="readonly")
        work_type_combobox.grid(row=i, column=1, sticky=tk.E, padx=5, pady=5)
        work_type_combobox.set(work_type_options[0])
        entries[label] = work_type_combobox
    elif label == 'Hypertension':
        hyper_tension_combobox = ttk.Combobox(frame, values=heart_disease_options, state="readonly")
        hyper_tension_combobox.grid(row=i, column=1, sticky=tk.E, padx=5, pady=5)
        hyper_tension_combobox.set(hypertension_options[0])
        entries[label] = hyper_tension_combobox
    else:
        entry = ttk.Entry(frame)
        entry.grid(row=i, column=1, sticky=tk.E, padx=5, pady=5)
        entries[label] = entry

# ... rest of the UI setup ...


# Adding the predict button
predict_button = ttk.Button(frame, text="Predict", command=on_predict)
predict_button.grid(row=len(labels), column=0, columnspan=2)

# Starting the application
root.mainloop()
