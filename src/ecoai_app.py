import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# -------------------------------
# 🎨 Page Configuration
# -------------------------------
st.set_page_config(
    page_title="EcoAI - Climate Predictor 🌍",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# 📂 Load Data
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data/GlobalLandTemperaturesByState.csv")
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.groupby(['dt', 'Country'])['AverageTemperature'].mean().reset_index()
    df = df.dropna(subset=['AverageTemperature'])
    # Keep only data after 1950
    df = df[df['dt'].dt.year >= 2000]
    return df

df = load_data()

# -------------------------------
# 🧭 Sidebar - Navigation
# -------------------------------
st.sidebar.title("🌍 EcoAI Climate Dashboard")
st.sidebar.write("Explore and forecast temperature trends across countries.")

# 🌍 Country Dropdown
countries = sorted(df['Country'].dropna().unique())
country_filter = st.sidebar.selectbox("Select a Country", countries, index=countries.index("India") if "India" in countries else 0)

future_months = st.sidebar.slider("Months to Forecast", 6, 60, 12, step=6)
show_raw = st.sidebar.checkbox("Show Raw Data", False)

# 📆 Year Range Slider
st.sidebar.markdown("### 📆 Select Year Range")
min_year = int(df['dt'].dt.year.min())
max_year = int(df['dt'].dt.year.max())
year_range = st.sidebar.slider("Year Range", min_year, max_year, (2000, max_year))

st.sidebar.markdown("---")
st.sidebar.info("📊 Tip: Use the sidebar to explore different countries and forecast durations.")

# -------------------------------
# 🌡️ Filter Data
# -------------------------------
df_country = df[df['Country'] == country_filter]
df_country = df_country[(df_country['dt'].dt.year >= year_range[0]) & (df_country['dt'].dt.year <= year_range[1])]

if df_country.empty:
    st.warning(f"No data found for '{country_filter}'. Try another country.")
    st.stop()

# -------------------------------
# 📈 Visualization: Historical Trends
# -------------------------------
st.title("🌡️ EcoAI Climate Forecast Dashboard")
st.markdown(f"### Historical Temperature Trends in **{country_filter}**")

fig = px.line(
    df_country, x='dt', y='AverageTemperature',
    title=f"Average Monthly Temperature - {country_filter}",
    labels={'AverageTemperature': 'Temperature (°C)', 'dt': 'Year'},
    template='plotly_dark'
)
st.plotly_chart(fig, use_container_width=True)

if show_raw:
    st.write(df_country.tail(10))

# -------------------------------
# 🤖 AI Forecasting (LSTM)
# -------------------------------
st.markdown("## 🔮 AI Forecast")

# Prepare data
df_series = df_country[['dt', 'AverageTemperature']].copy()
df_series['Month'] = df_series['dt'].dt.to_period('M')
df_monthly = df_series.groupby('Month')['AverageTemperature'].mean().reset_index()
df_monthly['Month'] = df_monthly['Month'].dt.to_timestamp()

scaler = MinMaxScaler()
scaled = scaler.fit_transform(np.array(df_monthly['AverageTemperature']).reshape(-1, 1))

look_back = 12
X, y = [], []
for i in range(len(scaled) - look_back):
    X.append(scaled[i:i+look_back])
    y.append(scaled[i+look_back])
X, y = np.array(X), np.array(y)

# Build Model
model = tf.keras.models.Sequential([
    tf.keras.layers.LSTM(64, return_sequences=False, input_shape=(look_back, 1)),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=10, batch_size=8, verbose=0)

# Predict future
future_preds = []
last_sequence = scaled[-look_back:]
for _ in range(future_months):
    pred = model.predict(last_sequence.reshape(1, look_back, 1), verbose=0)
    future_preds.append(pred[0, 0])
    last_sequence = np.append(last_sequence[1:], pred).reshape(look_back, 1)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
future_dates = pd.date_range(df_monthly['Month'].iloc[-1] + pd.offsets.MonthBegin(), periods=future_months, freq='MS')
df_future = pd.DataFrame({'Month': future_dates, 'PredictedTemperature': future_preds.flatten()})

# -------------------------------
# 🌤️ Plot Forecast (Color Enhanced)
# -------------------------------
fig2 = go.Figure()

# Historical Line (Blue)
fig2.add_trace(go.Scatter(
    x=df_monthly['Month'],
    y=df_monthly['AverageTemperature'],
    mode='lines',
    name='Historical',
    line=dict(color='deepskyblue', width=2)
))

# Forecast Line (Orange Dashed)
fig2.add_trace(go.Scatter(
    x=df_future['Month'],
    y=df_future['PredictedTemperature'],
    mode='lines',
    name='AI Forecast',
    line=dict(color='orange', width=3, dash='dash')
))

# Highlight forecast area (optional)
fig2.add_vrect(
    x0=df_future['Month'].iloc[0],
    x1=df_future['Month'].iloc[-1],
    fillcolor="orange", opacity=0.1, line_width=0
)

fig2.update_layout(
    title=f"AI Forecast for Next {future_months} Months in {country_filter}",
    xaxis_title="Date",
    yaxis_title="Temperature (°C)",
    template='plotly_dark',
    showlegend=True
)

st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# 🧾 Summary Box
# -------------------------------
latest_temp = round(df_monthly['AverageTemperature'].iloc[-1], 2)
forecast_end = round(df_future['PredictedTemperature'].iloc[-1], 2)
temp_change = forecast_end - latest_temp

st.markdown("### 📊 Forecast Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Current Avg Temp", f"{latest_temp}°C")
col2.metric(f"Forecast ({future_months} mo)", f"{forecast_end}°C")
col3.metric("Change", f"{temp_change:+.2f}°C")

st.success(
    f"EcoAI predicts that average temperature in **{country_filter}** may change by **{temp_change:+.2f}°C** "
    f"over the next **{future_months} months**."
)
