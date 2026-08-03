import pickle
import gradio as gr
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
with open("D:/DataScience/Lab06_Logistic_Regression/Script/random_forest_model.pkl", "rb") as model_file:
    rf_model = pickle.load(model_file)

# Define a function to predict diabetes
def predict_diabetes(pregnancies, glucose, bmi, age):
    # Convert input into a numpy array
    input_data = np.array([[pregnancies, glucose, bmi, age]])
    
    # Get class prediction
    prediction = rf_model.predict(input_data)[0]
    
    # Get probability predictions for both classes
    probabilities = rf_model.predict_proba(input_data)[0] * 100  # Convert to percentage
    
    # Label the result
    result_label = "Diabetic (1)" if prediction == 1 else "Non-Diabetic (0)"
    
    # return result_label, probabilities[0], probabilities[1] 
    # Create pie chart
    fig, ax = plt.subplots(figsize=(4, 2.15)) 
    labels = ['Non-Diabetic', 'Diabetic']  #52525b
    colors = ['#52525b', '#eb5c0c']   
    explode = (0.1, 0) if prediction == 0 else (0, 0.1)  # explode the predicted class
    
    ax.pie(probabilities, 
           labels=labels, 
           autopct='%1.1f%%',
           colors=colors,
           startangle=90,
           explode=explode,
           shadow=True)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    return result_label, fig

# Create Gradio Interface
interface = gr.Interface(
    fn=predict_diabetes,
    inputs=[
        gr.Slider(0, 20, step=1, label="Pregnancies"),
        gr.Number(label="Glucose (mg/dL)"),
        gr.Number(label="BMI (kg/m²)"),
        gr.Slider(20, 100, step=1, label="Age (years)")
    ],
    outputs=[
        gr.Textbox(label="Prediction Output"), 
        gr.Plot(label="Prediction Probabilities")
    ],
    title="Diabetes Prediction",
    description="Enter the values and predict whether a person is diabetic or not.",
    live=False
)

# Launch Gradio app
interface.launch()
