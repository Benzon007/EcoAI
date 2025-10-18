# =============================
# EcoAI: Climate Data Predictor
# =============================

# Import libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

# =============================
# Step 1: Load and Prepare Data
# =============================

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# Load dataset (change path if needed)
df = pd.read_csv("data/GlobalLandTemperaturesByState.csv")

# Convert date column to datetime
df['dt'] = pd.to_datetime(df['dt'])

# Drop missing values
df = df.dropna(subset=['AverageTemperature'])

# Filter for a specific country (e.g., India)
country = 'India'
df_country = df[df['Country'] == country]

# Group by date and calculate monthly mean
df_monthly = (
    df_country.groupby('dt')['AverageTemperature']
    .mean()
    .reset_index()
    .sort_values('dt')
)

# Keep only data after 1900 for clarity
df_monthly = df_monthly[df_monthly['dt'].dt.year >= 1900]

# =============================
# Step 2: Visualize Historical Trend
# =============================

plt.figure(figsize=(10, 4))
plt.plot(df_monthly['dt'], df_monthly['AverageTemperature'])
plt.title(f"{country} - Monthly Mean Temperature")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.tight_layout()
plt.show()

# =============================
# Step 3: Prepare Data for AI Model
# =============================

# Convert temperature to numpy array
series = df_monthly['AverageTemperature'].values.reshape(-1, 1)

# Scale data between 0 and 1
scaler = MinMaxScaler()
series_scaled = scaler.fit_transform(series)

# Function to create time window sequences for LSTM
def create_sequences(data, window_size=24):
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size])
    return np.array(X), np.array(y)

WINDOW_SIZE = 24  # 24 months = 2 years of history
X, y = create_sequences(series_scaled, WINDOW_SIZE)

# Train-test split
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

print(f"Training samples: {len(X_train)}, Testing samples: {len(X_test)}")

# =============================
# Step 4: Build and Train LSTM Model
# =============================

model = tf.keras.Sequential([
    tf.keras.layers.LSTM(64, input_shape=(X_train.shape[1], X_train.shape[2])),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

print("\nTraining model...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=25,
    batch_size=16,
    verbose=1
)

# =============================
# Step 5: Evaluate & Predict
# =============================

pred_scaled = model.predict(X_test)
pred = scaler.inverse_transform(pred_scaled)
y_true = scaler.inverse_transform(y_test)

mse = mean_squared_error(y_true, pred)
mae = mean_absolute_error(y_true, pred)
print(f"\nModel Performance: MSE = {mse:.4f}, MAE = {mae:.4f}")

plt.figure(figsize=(10, 4))
plt.plot(y_true, label="Actual Temperature")
plt.plot(pred, linestyle='--', label="Predicted Temperature")
plt.title(f"{country} - EcoAI Temperature Forecast")
plt.xlabel("Time Step (Months)")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()

# =============================
# Step 6: Forecast Next 12 Months
# =============================

future_steps = 12  # number of months to forecast
last_sequence = series_scaled[-WINDOW_SIZE:].reshape(1, WINDOW_SIZE, 1)

future_preds = []
for _ in range(future_steps):
    next_pred = model.predict(last_sequence)[0][0]
    future_preds.append(next_pred)
    # Append prediction and slide window forward
    last_sequence = np.append(last_sequence[:, 1:, :], [[[next_pred]]], axis=1)

# Inverse transform predictions
future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

# Create future dates
last_date = df_monthly['dt'].iloc[-1]
future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1),
                             periods=future_steps, freq='MS')

# Build future forecast DataFrame
df_future = pd.DataFrame({
    'Date': future_dates,
    'PredictedTemperature': future_preds.flatten()
})

# Plot forecast continuation
plt.figure(figsize=(12, 5))
plt.plot(df_monthly['dt'][-120:], df_monthly['AverageTemperature'][-120:], label='Past Temperature')
plt.plot(df_future['Date'], df_future['PredictedTemperature'], 'r--', label='Future Forecast')
plt.title(f"{country} - Next 12-Month Temperature Forecast (EcoAI)")
plt.xlabel("Date")
plt.ylabel("Temperature (°C)")
plt.legend()
plt.tight_layout()
plt.show()

# Save forecast CSV
df_future.to_csv("outputs/ecoai_future_forecast.csv", index=False)
print("\nSaved 12-month forecast to outputs/ecoai_future_forecast.csv")
print(df_future)

# =============================
# Step 7: Combine Historical + Forecast for Tableau
# =============================

df_past = df_monthly[['dt', 'AverageTemperature']].rename(
    columns={'dt': 'Date', 'AverageTemperature': 'Temperature'}
)
df_future = df_future.rename(columns={'PredictedTemperature': 'Temperature'})
df_future['Type'] = 'Forecast'
df_past['Type'] = 'Historical'

# Merge and save combined dataset
df_all = pd.concat([df_past, df_future], ignore_index=True)
df_all.to_csv("outputs/ecoai_tableau_data.csv", index=False)
print("✅ Saved combined dataset for Tableau: outputs/ecoai_tableau_data.csv")
