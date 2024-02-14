!pip install flask
!pip install flask-ngrok
!pip install pyngrok
#MORE THAN ONE ROW IN A FILE
import os
import pandas as pd
from flask import Flask, request, render_template
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
import logging

# Set up logging
logging.basicConfig(filename='app.log', level=logging.DEBUG)

# Load the CSV data into a Pandas DataFrame
data = pd.read_csv('bibertrain.csv')

# Extract features (numerical frequencies) and target (communicative purpose)
X_train = data.iloc[:, 3:]  # Assuming feature columns start from the fourth column
y_train = LabelEncoder().fit_transform(data['main_full'])

# Create and train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Create and fit a label encoder for inverse transformation
label_encoder = LabelEncoder()
label_encoder.fit(data['main_full'])

# Create and fit a scaler for scaling input data
scaler = StandardScaler()
scaler.fit(X_train)

# Initialize Flask app
app = Flask(__name__, template_folder='./')

# Define route for prediction
@app.route('/', methods=['GET', 'POST'])
def predict_communicative_purpose():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        if uploaded_file:
            try:
                input_data = pd.read_csv(uploaded_file)

                # Ensure the order of columns in the input data matches the training data
                input_data = input_data[X_train.columns]

                # Scale the input features using the same scaler
                input_data_scaled = scaler.transform(input_data)

                # Make predictions for each row in the input data
                predictions = []
                for i in range(len(input_data_scaled)):
                    row_input = input_data_scaled[i].reshape(1, -1)
                    prediction_probabilities = model.predict_proba(row_input)[0]
                    top_communicative_purpose_index = prediction_probabilities.argmax()
                    top_communicative_purpose = label_encoder.inverse_transform([top_communicative_purpose_index])[0]
                    top_probability = prediction_probabilities[top_communicative_purpose_index]
                    percentages = {label: probability * 100 for label, probability in zip(model.classes_, prediction_probabilities)}
                    predictions.append({
                        'communicative_purpose': top_communicative_purpose,
                        'probability': top_probability,
                        'percentages': percentages
                    })

                logging.debug("Input Data:")
                logging.debug(input_data)
                logging.debug(f"Predictions: {predictions}")

                return render_template('index.html', predictions=predictions)
            except Exception as e:
                logging.error(f"Prediction Error: {str(e)}")
                return render_template('index.html', error_message=str(e))

    return render_template('index.html', predictions=None, error_message=None)

# Run the Flask app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5007)
