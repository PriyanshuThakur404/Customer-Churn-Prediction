# Customer Churn Prediction Web Application

This is a web application that predicts customer churn using machine learning. The application uses Flask for the backend, a Random Forest Classifier for predictions, and a simple HTML/CSS frontend.

## Features

- Machine learning model for churn prediction
- Web interface for inputting customer data
- Real-time prediction results
- Probability display for predictions
- Input validation
- Responsive design using Bootstrap

## Setup Instructions

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
python app.py
```

4. Open your web browser and go to `http://localhost:5000`

## Usage

1. Enter the customer's information:
   - Tenure (in months)
   - Monthly Charges
   - Total Charges

2. Click "Predict Churn" to get the prediction

3. The result will show:
   - Prediction (Churn or Not Churn)
   - Probability of churn

## Project Structure

- `app.py` - Main Flask application
- `templates/` - HTML templates
- `model/` - Directory for storing the trained model
- `requirements.txt` - Project dependencies

## Note

This application uses a sample dataset for demonstration purposes. In a real-world scenario, you should:
1. Use a larger, more comprehensive dataset
2. Perform proper data preprocessing
3. Implement more sophisticated feature engineering
4. Use cross-validation for model evaluation
5. Consider implementing model retraining capabilities 