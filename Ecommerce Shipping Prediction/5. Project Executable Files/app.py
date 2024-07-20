from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, Normalizer
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)

# Load models and encoders
model = joblib.load('RandomForest_best_model.pkl')
Data_normalizer = joblib.load('Normalizer.pkl')

# Label encoders
label_encoders = {
    'Warehouse_block': joblib.load('label_encoder_Warehouse_block.pkl'),
    'Mode_of_Shipment': joblib.load('label_encoder_Mode_of_Shipment.pkl'),
    'Product_importance': joblib.load('label_encoder_Product_importance.pkl'),
    'Gender': joblib.load('label_encoder_Gender.pkl')
}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Extract data from form
        Warehouse_block = request.form['warehouse_block']
        Mode_of_Shipment = request.form['mode_of_shipment']
        Customer_care_calls = int(request.form['customer_care_calls'])
        Customer_rating = int(request.form['customer_rating'])
        Cost_of_the_Product = float(request.form['cost_of_product'])
        Prior_purchases = int(request.form['prior_purchases'])
        Product_importance = request.form['product_importance']
        Gender = request.form['gender']
        Discount_offered = float(request.form['discount_offered'])
        Weight_in_gms = float(request.form['weight_in_gms'])

        # Convert categorical data to numeric values using label encoders
        Warehouse_block = label_encoders['Warehouse_block'].transform([Warehouse_block])[0]
        Mode_of_Shipment = label_encoders['Mode_of_Shipment'].transform([Mode_of_Shipment])[0]
        Product_importance = label_encoders['Product_importance'].transform([Product_importance])[0]
        Gender = label_encoders['Gender'].transform([Gender])[0]

        # Create a list of the features
        preds = [[Warehouse_block, Mode_of_Shipment, Customer_care_calls, Customer_rating, 
                  Cost_of_the_Product, Prior_purchases, Product_importance, Gender, 
                  Discount_offered, Weight_in_gms]]

        # Normalize the data
        preds = Data_normalizer.transform(preds)

        # Predict the probability
        prob = model.predict_proba(preds)[0][1]  # Assuming the second column is the probability of interest

        return render_template('predict.html', probability=round(prob * 100, 2))
    return render_template('predict.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)