# Bengaluru Home Price Estimator

Streamlit app that predicts house prices in Bengaluru using a Linear Regression model.

Files:
- model_training.ipynb - data cleaning and model training
- app.py - the streamlit app
- price_model.pkl - trained model
- column_transformer.pkl - preprocessing pipeline
- requirements.txt - dependencies

Setup:
pip install -r requirements.txt

Run:
streamlit run app.py

Make sure app.py, price_model.pkl and column_transformer.pkl are in the same folder.

Model: Linear Regression, R2 ~0.844 on test data. Price is log-transformed during training and converted back for predictions.
This data was taken from https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data.