from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)

# Initialize database
def init_db():
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS predictions
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  date TEXT,
                  customer_id TEXT,
                  prediction TEXT,
                  probability REAL,
                  contract_type TEXT,
                  monthly_charges REAL,
                  tenure INTEGER,
                  internet_service TEXT,
                  online_security TEXT,
                  tech_support TEXT,
                  customer_service_calls INTEGER,
                  late_payments INTEGER,
                  status TEXT DEFAULT 'pending')''')
    conn.commit()
    conn.close()

init_db()

# Create model directory if it doesn't exist
if not os.path.exists('model'):
    os.makedirs('model')

# Function to train and save the model
def train_model():
    # Create a larger sample dataset with more features
    np.random.seed(42)
    n_samples = 2000  # Increased sample size
    
    # Generate synthetic data with more realistic patterns
    data = {
        # Basic customer information
        'tenure': np.random.randint(1, 72, n_samples),  # 1-72 months
        'monthly_charges': np.random.uniform(20, 120, n_samples),
        'total_charges': np.random.uniform(50, 5000, n_samples),
        
        # Contract and service information
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.5, 0.3, 0.2]),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n_samples),
        
        # Additional service features
        'phone_service': np.random.choice(['Yes', 'No'], n_samples, p=[0.8, 0.2]),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.4, 0.2]),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'online_backup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'device_protection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'streaming_movies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        
        # Customer behavior metrics
        'avg_monthly_usage_gb': np.random.uniform(0, 1000, n_samples),
        'customer_service_calls': np.random.randint(0, 10, n_samples),
        'late_payments': np.random.randint(0, 5, n_samples),
        
        # Generate churn with more realistic patterns
        'churn': np.zeros(n_samples)
    }
    
    # Create more realistic churn patterns based on features
    for i in range(n_samples):
        churn_prob = 0.0
        
        # Higher churn for month-to-month contracts
        if data['contract_type'][i] == 'Month-to-month':
            churn_prob += 0.3
        
        # Higher churn for higher monthly charges
        if data['monthly_charges'][i] > 80:
            churn_prob += 0.2
        
        # Higher churn for more customer service calls
        churn_prob += data['customer_service_calls'][i] * 0.05
        
        # Higher churn for late payments
        churn_prob += data['late_payments'][i] * 0.1
        
        # Lower churn for longer tenure
        churn_prob -= min(data['tenure'][i] * 0.01, 0.2)
        
        # Add some randomness
        churn_prob += np.random.uniform(-0.1, 0.1)
        
        # Ensure probability is between 0 and 1
        churn_prob = max(0, min(1, churn_prob))
        
        # Set churn based on probability
        data['churn'][i] = 1 if np.random.random() < churn_prob else 0
    
    df = pd.DataFrame(data)
    
    # Convert categorical variables to numerical
    categorical_columns = [
        'contract_type', 'internet_service', 'payment_method',
        'phone_service', 'multiple_lines', 'online_security',
        'online_backup', 'device_protection', 'tech_support',
        'streaming_tv', 'streaming_movies'
    ]
    df = pd.get_dummies(df, columns=categorical_columns)
    
    # Prepare features and target
    X = df.drop('churn', axis=1)
    y = df['churn']
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the model with more trees
    model = RandomForestClassifier(n_estimators=200, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, 'model/churn_model.joblib')
    return model

# Load or train the model
model_path = 'model/churn_model.joblib'
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    model = train_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        tenure = int(request.form['tenure'])
        monthly_charges = float(request.form['monthly_charges'])
        total_charges = float(request.form['total_charges'])
        contract_type = request.form['contract_type']
        internet_service = request.form['internet_service']
        payment_method = request.form['payment_method']
        phone_service = request.form['phone_service']
        multiple_lines = request.form['multiple_lines']
        online_security = request.form['online_security']
        online_backup = request.form['online_backup']
        device_protection = request.form['device_protection']
        tech_support = request.form['tech_support']
        streaming_tv = request.form['streaming_tv']
        streaming_movies = request.form['streaming_movies']
        avg_monthly_usage_gb = float(request.form['avg_monthly_usage_gb'])
        customer_service_calls = int(request.form['customer_service_calls'])
        late_payments = int(request.form['late_payments'])

        # Create input data with categorical variables
        input_data = pd.DataFrame({
            'tenure': [tenure],
            'monthly_charges': [monthly_charges],
            'total_charges': [total_charges],
            'contract_type': [contract_type],
            'internet_service': [internet_service],
            'payment_method': [payment_method],
            'phone_service': [phone_service],
            'multiple_lines': [multiple_lines],
            'online_security': [online_security],
            'online_backup': [online_backup],
            'device_protection': [device_protection],
            'tech_support': [tech_support],
            'streaming_tv': [streaming_tv],
            'streaming_movies': [streaming_movies],
            'avg_monthly_usage_gb': [avg_monthly_usage_gb],
            'customer_service_calls': [customer_service_calls],
            'late_payments': [late_payments]
        })

        # Convert categorical variables to one-hot encoding
        categorical_columns = [
            'contract_type', 'internet_service', 'payment_method',
            'phone_service', 'multiple_lines', 'online_security',
            'online_backup', 'device_protection', 'tech_support',
            'streaming_tv', 'streaming_movies'
        ]
        
        # Create dummy variables
        input_data = pd.get_dummies(input_data, columns=categorical_columns)
        
        # Ensure all columns from training are present
        for col in model.feature_names_in_:
            if col not in input_data.columns:
                input_data[col] = 0

        # Reorder columns to match training data
        input_data = input_data[model.feature_names_in_]

        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1] * 100

        # Generate a unique customer ID
        customer_id = f"CUST-{datetime.now().strftime('%Y%m%d%H%M%S')}"

        # Store prediction in database
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('''INSERT INTO predictions 
                    (date, customer_id, prediction, probability, contract_type, 
                     monthly_charges, tenure, internet_service, online_security, 
                     tech_support, customer_service_calls, late_payments)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                 (datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                  customer_id,
                  'Churn' if prediction == 1 else 'Not Churn',
                  probability,
                  contract_type,
                  monthly_charges,
                  tenure,
                  internet_service,
                  online_security,
                  tech_support,
                  customer_service_calls,
                  late_payments))
        conn.commit()
        conn.close()

        # Prepare key factors for display
        key_factors = {
            'contract_type': contract_type,
            'monthly_charges': monthly_charges,
            'tenure': tenure,
            'internet_service': internet_service,
            'online_security': online_security,
            'tech_support': tech_support,
            'customer_service_calls': customer_service_calls,
            'late_payments': late_payments
        }

        result = {
            'prediction': 'Churn' if prediction == 1 else 'Not Churn',
            'probability': round(probability, 2),
            'factors': key_factors,
            'customer_id': customer_id
        }

        return render_template('index.html', result=result)

    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/get_history', methods=['GET'])
def get_history():
    try:
        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('SELECT * FROM predictions ORDER BY date DESC')
        predictions = c.fetchall()
        conn.close()

        history = []
        for pred in predictions:
            history.append({
                'id': pred[0],
                'date': pred[1],
                'customer_id': pred[2],
                'prediction': pred[3],
                'probability': pred[4],
                'contract_type': pred[5],
                'monthly_charges': pred[6],
                'tenure': pred[7],
                'internet_service': pred[8],
                'online_security': pred[9],
                'tech_support': pred[10],
                'customer_service_calls': pred[11],
                'late_payments': pred[12],
                'status': pred[13]
            })

        return jsonify(history)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_status', methods=['POST'])
def update_status():
    try:
        data = request.json
        prediction_id = data.get('id')
        new_status = data.get('status')

        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('UPDATE predictions SET status = ? WHERE id = ?', (new_status, prediction_id))
        conn.commit()
        conn.close()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/delete_prediction', methods=['POST'])
def delete_prediction():
    try:
        data = request.json
        prediction_id = data.get('id')

        conn = sqlite3.connect('predictions.db')
        c = conn.cursor()
        c.execute('DELETE FROM predictions WHERE id = ?', (prediction_id,))
        conn.commit()
        conn.close()

        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080) 