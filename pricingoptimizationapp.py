# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# Load a predefined dataset (you can replace this with your dataset)
@st.cache
def load_sample_data():
    # Generating synthetic data for demonstration purposes
    # This could be replaced by your actual dataset or a small sample of it
    date_range = pd.date_range(start='2020-01-01', periods=1000, freq='H')
    data = pd.DataFrame({
        'timestamp': date_range,
        'total_wind_production': np.sin(np.linspace(0, 100, 1000)) + np.random.normal(0, 0.1, 1000),
        'temperature': np.random.uniform(0, 30, 1000),
        'wind_speed': np.random.uniform(0, 15, 1000),
    })
    return data

# Load your trained LSTM model (you can save it from Colab and then load it in the Streamlit app)
@st.cache(allow_output_mutation=True)
def load_lstm_model():
    model = load_model('lstm_model.keras')  # Provide the path to your saved LSTM model
    return model

# Helper function to create sequences for LSTM input
def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data) - seq_length):
        x = data[i:i+seq_length, :-1]  # All features except the target (last column)
        y = data[i+seq_length, -1]  # The target is the last column (wind energy production)
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

# Preprocess the data and ensure it's correctly shaped for LSTM
X_lstm, y_lstm = create_sequences(scaled_data, sequence_length)

# The shape of X_lstm should be (num_samples, sequence_length, num_features)
print(X_lstm.shape)  # Ensure this prints something like (n_samples, sequence_length, n_features)

# App Title
st.title("Wind Energy Production Prediction")

# Sidebar for user inputs
st.sidebar.title("Model Settings")
sequence_length = st.sidebar.slider("Sequence Length", min_value=10, max_value=100, value=60, step=5)

# Load the predefined dataset
data = load_sample_data()

# Show the data in the app
st.write("Dataset Preview")
st.dataframe(data.head())

# Preprocessing: scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['total_wind_production', 'temperature', 'wind_speed']])

# Create sequences
X_lstm = create_sequences(scaled_data, sequence_length)

# Load the LSTM model
lstm_model = load_lstm_model()

# Predict on the preloaded data
predictions = lstm_model.predict(X_lstm)

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

# Footer with instructions
st.sidebar.subheader("Instructions")
st.sidebar.write("""
- Adjust the model settings using the sidebar.
- View predictions and performance metrics in the main area.
""")
