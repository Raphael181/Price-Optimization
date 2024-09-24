# Import necessary libraries
%%writefile priceoptimization.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load your trained LSTM model (you can save it from Colab and then load it in the Streamlit app)
@st.cache(allow_output_mutation=True)
def load_lstm_model():
    model = load_model('your_lstm_model.h5')  # Provide the path to your saved LSTM model
    return model

# Helper function to create sequences for LSTM input
def create_sequences(data, seq_length):
    xs = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]  # All features except target (total wind production)
        xs.append(x)
    return np.array(xs)

# App Title
st.title("Wind Energy Production Prediction")

# Sidebar for user inputs
st.sidebar.title("Model Settings")
sequence_length = st.sidebar.slider("Sequence Length", min_value=10, max_value=100, value=60, step=5)

# File upload for dataset
uploaded_file = st.sidebar.file_uploader("Upload your CSV dataset", type=["csv"])
if uploaded_file is not None:
    # Load the dataset
    data = pd.read_csv(uploaded_file)
    
    # Display a message when the file is uploaded
    st.sidebar.success("File uploaded successfully!")
    
    # Show the data in the app
    st.write("Dataset Preview")
    st.dataframe(data.head())

    # Preprocessing: scaling
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    # Create sequences
    X_lstm = create_sequences(scaled_data, sequence_length)
    
    # Load the LSTM model
    lstm_model = load_lstm_model()

    # Predict on the uploaded data
    predictions = lstm_model.predict(X_lstm)
    
    # Inverse transform the predictions
    predictions_rescaled = scaler.inverse_transform(
        np.concatenate([X_lstm[:, -1], predictions.reshape(-1, 1)], axis=1))[:, -1]

    # Visualize predictions vs actual values
    st.subheader("Wind Energy Production Predictions")
    df_results = pd.DataFrame({
        'Actual': data['total_wind_production'][sequence_length:],
        'Predicted': predictions_rescaled
    })
    
    # Plot results
    fig = px.line(df_results, title="Actual vs Predicted Wind Energy Production")
    st.plotly_chart(fig)

    # Display prediction metrics (MSE, R²)
    st.subheader("Model Performance")
    mse = np.mean((df_results['Actual'] - df_results['Predicted'])**2)
    r2 = 1 - (np.sum((df_results['Actual'] - df_results['Predicted'])**2) / np.sum((df_results['Actual'] - np.mean(df_results['Actual']))**2))
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**R-squared (R²):** {r2:.4f}")

else:
    st.sidebar.warning("Please upload a CSV file to proceed.")

# Footer with instructions
st.sidebar.subheader("Instructions")
st.sidebar.write("""
- Upload a CSV file with wind production data.
- Adjust the model settings using the sidebar.
- View predictions and performance metrics in the main area.
""")
