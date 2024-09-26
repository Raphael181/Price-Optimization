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

# Streamlit App Layout

st.title("Wind Energy Production Forecasting")
st.markdown("### An interactive app to predict wind energy production using an LSTM model.")

# Step 1: User Inputs for Model Parameters
st.sidebar.header("Model Settings")
seq_length = st.sidebar.slider('Sequence Length for LSTM Input', min_value=10, max_value=100, value=60, step=10)
lstm_units = st.sidebar.slider('Number of LSTM Units', min_value=10, max_value=200, value=50, step=10)
epochs = st.sidebar.slider('Number of Epochs', min_value=1, max_value=50, value=10, step=1)

# Step 2: Feature Selection
st.sidebar.header("Select Features")
feature_columns = ['total_wind_production', 'temperature', 'wind_speed']
selected_features = st.sidebar.multiselect("Select Features to Include", options=feature_columns, default=feature_columns)

# Step 3: Load and Preprocess the Dataset
data = load_sample_data()
st.write("Dataset Preview")
st.dataframe(data.head())

# Ensure the selected features are valid for scaling
if len(selected_features) > 0:
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data[selected_features])
    
    # Step 4: Create sequences for LSTM input
    X_lstm, y_lstm = create_sequences(scaled_data, seq_length)

    # Step 5: Define and Train the LSTM Model
    model = Sequential()
    model.add(LSTM(units=lstm_units, return_sequences=False, input_shape=(seq_length, X_lstm.shape[2])))
    model.add(Dense(1))  # Output layer with 1 neuron for the target variable
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    st.write(f"Training the model with {epochs} epochs...")
    model.fit(X_lstm, y_lstm, epochs=epochs, batch_size=32, validation_split=0.2, verbose=2)

    # Step 6: Make Predictions
    predictions = model.predict(X_lstm)

    # Inverse transform the predictions if necessary
    if 'total_wind_production' in selected_features:
        last_index = selected_features.index('total_wind_production')
        predictions_rescaled = scaler.inverse_transform(
            np.concatenate([X_lstm[:, -1], predictions.reshape(-1, 1)], axis=1))[:, -1]
    else:
        predictions_rescaled = predictions.flatten()  # No need to inverse transform if 'total_wind_production' not in features

    # Step 7: Visualizations
    st.subheader("Wind Energy Production Predictions")
    df_results = pd.DataFrame({
        'Actual': data['total_wind_production'][seq_length:].values,
        'Predicted': predictions_rescaled
    })

    # Interactive Plot
    fig = px.line(df_results, title="Actual vs Predicted Wind Energy Production", markers=True)
    fig.update_layout(hovermode="x")
    st.plotly_chart(fig)

    # Step 8: Display Metrics
    st.subheader("Model Performance")
    mse = np.mean((df_results['Actual'] - df_results['Predicted'])**2)
    r2 = 1 - (np.sum((df_results['Actual'] - df_results['Predicted'])**2) / np.sum((df_results['Actual'] - np.mean(df_results['Actual']))**2))
    st.write(f"**Mean Squared Error (MSE):** {mse:.4f}")
    st.write(f"**R-squared (R²):** {r2:.4f}")

    # Step 9: Download Predictions
    st.subheader("Download Predictions")
    csv = df_results.to_csv(index=False)
    st.download_button(label="Download CSV", data=csv, file_name='predictions.csv', mime='text/csv')

else:
    st.write("Please select at least one feature for scaling and prediction.")
