
# Diabetes Prediction App

This project is a Streamlit-based web application for predicting the likelihood of diabetes based on health parameters. The app takes user inputs, such as glucose levels, BMI, and age, and uses a machine learning model to predict the probability of diabetes.

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [File Structure](#file-structure)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

This app uses an XGBoost model trained on a diabetes dataset to predict whether a person is likely to have diabetes. The model was fine-tuned using Optuna for hyperparameter optimization, resulting in a high-performance predictive model.

## Features

- **User-friendly Interface**: Simple and interactive input form for user data.
- **Real-time Prediction**: Immediate feedback on diabetes likelihood after data input.
- **Optimized Model**: The app leverages an XGBoost model fine-tuned for accuracy.

## Installation

To run this app, you'll need to have Python installed along with some dependencies. Follow these steps to set up and run the app:

1. **Clone the Repository** (or download it directly):

   ```bash
   git clone https://github.com/yourusername/diabetes-prediction-app.git
   cd diabetes-prediction-app
   ```

2. **Install Dependencies**:

   Use `pip` to install required libraries.

   ```bash
   pip install -r requirements.txt
   ```

3. **Download the Model**:

   Make sure the `diabetes_xgb_model.pkl` file is in the app's root directory. This file is required for the app to make predictions.

## Usage

1. Run the app using Streamlit:

   ```bash
   streamlit run diabetes_app.py
   ```

2. Open the local URL (usually `http://localhost:8501`) in your browser.

3. Enter the required details:
   - **Pregnancies**: Number of times pregnant
   - **Glucose Level**: Plasma glucose concentration
   - **Blood Pressure**: Diastolic blood pressure
   - **Skin Thickness**
   - **Insulin**: Serum insulin
   - **BMI**: Body Mass Index
   - **Diabetes Pedigree Function**: Genetic impact indicator
   - **Age**

4. Click **Predict Diabetes** to see the prediction.

## Model Training

The XGBoost model was trained using the diabetes dataset. The model was fine-tuned using Optuna to maximize prediction accuracy. Training details and code are in the `diabetis.ipynb` Jupyter notebook file.

To retrain or fine-tune the model, run `diabetis.ipynb`, save the best-performing model, and update `diabetes_xgb_model.pkl` in the app directory.

## File Structure

- `diabetes_app.py`: The main Streamlit app file.
- `diabetis.ipynb`: Notebook for data analysis, model training, and tuning.
- `diabetes_xgb_model.pkl`: Trained XGBoost model saved as a pickle file.
- `requirements.txt`: List of dependencies.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for feedback.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
