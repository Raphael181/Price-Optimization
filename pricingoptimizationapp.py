# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# Load a predefined dataset (synthetic for demonstration purposes)
@st.cache
def load_sample_data():
    # Generating synthetic data for demonstration purposes
    date_range = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    data = pd.DataFrame({
        'timestamp': date_range,
        'total_wind_production': np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000),
        'temperature': np.random.uniform(0, 30, 1000),
        'wind_speed': np.random.uniform(0, 15, 1000),
    })
    return data

# Helper function to create sequences for LSTM input
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]  # All features except target (last column)
        y = data[i+seq_length, -1]  # The target is the last column (wind energy production)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Load and preprocess the dataset
data = load_sample_data()

# Show the data in the app
st.write("Dataset Preview")
st.dataframe(data.head())

# Feature Scaling (Min-Max Scaling)
scaler = MinMaxScaler()

# Select the columns to be scaled
scaled_data = scaler.fit_transform(data[['total_wind_production', 'temperature', 'wind_speed']])

# Create sequences from the scaled data
sequence_length = 60  # Choose a sequence length for LSTM
X_lstm, y_lstm = create_sequences(scaled_data, sequence_length)

# The shape of X_lstm should be (num_samples, sequence_length, num_features)
st.write(f"Shape of input data (X): {X_lstm.shape}")
st.write(f"Shape of target data (y): {y_lstm.shape}")

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=False, input_shape=(sequence_length, X_lstm.shape[2])))
model.add(Dense(1))  # Output layer with 1 neuron for the target variable

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_lstm, y_lstm, epochs=10, batch_size=32, validation_split=0.2)

# Predict on the preloaded data
predictions = model.predict(X_lstm)

# Inverse transform the predictions
predictions_rescaled = scaler.inverse_transform(
    np.concatenate([X_lstm[:, -1], predictions.reshape(-1, 1)], axis=1))[:, -1]

# Visualize predictions vs actual values
st.subheader("Wind Energy Production Predictions")
df_results = pd.DataFrame({
    'Actual': data['total_wind_production'][sequence_length:].values,
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
