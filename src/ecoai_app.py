import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gc


# TensorFlow runtime tuning (safe mode)
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass  # Streamlit Cloud restricts thread configuration, safe to ignore


# Page Configuration
st.set_page_config(
    page_title="EcoAI - Climate Predictor 🌍",
    page_icon="🌡️",
    layout="wide",
    initial_sidebar_state="expanded"
)



# Load Data (cached)
@st.cache_data(ttl=86400)  # cache for 1 day
def load_data():
    df = pd.read_csv("data/GlobalLandTemperaturesByState.csv")
    df['dt'] = pd.to_datetime(df['dt'])
    df = df.groupby(['dt', 'Country'])['AverageTemperature'].mean().reset_index()
    df = df.dropna(subset=['AverageTemperature'])
    df = df[df['dt'].dt.year >= 1950]
    return df

df = load_data()


# Sidebar
st.sidebar.title("🌍 EcoAI Climate Dashboard")
st.sidebar.write("Explore and forecast temperature trends across countries.")

countries = sorted(df['Country'].dropna().unique())
country_filter = st.sidebar.selectbox(
    "Select a Country",
    countries,
    index=countries.index("India") if "India" in countries else 0
)

future_months = st.sidebar.slider("Months to Forecast", 6, 60, 12, step=6)
show_raw = st.sidebar.checkbox("Show Raw Data", False)

min_year = int(df['dt'].dt.year.min())
max_year = int(df['dt'].dt.year.max())
year_range = st.sidebar.slider("Year Range", min_year, max_year, (1950, max_year))

st.sidebar.markdown("---")
st.sidebar.info("⚡ Optimized for Streamlit Cloud performance.")


# Filter & Clean Data
df_country = df[df['Country'] == country_filter].copy()
df_country = df_country[
    (df_country['dt'].dt.year >= year_range[0]) &
    (df_country['dt'].dt.year <= year_range[1])
]

if df_country.empty:
    st.warning(f"No data found for '{country_filter}'. Try another country.")
    st.stop()

df_country['AverageTemperature'] = df_country['AverageTemperature'].interpolate('linear')
df_country['SmoothedTemp'] = df_country['AverageTemperature'].rolling(window=3, min_periods=1).mean()


# Visualization: Historical Trends
st.title("🌡️ EcoAI Climate Forecast Dashboard")
st.markdown(f"### Historical Temperature Trends in **{country_filter}**")

fig = px.line(
    df_country, x='dt', y='SmoothedTemp',
    title=f"Average Monthly Temperature - {country_filter}",
    labels={'SmoothedTemp': 'Temperature (°C)', 'dt': 'Year'},
    template='plotly_dark'
)
st.plotly_chart(fig, use_container_width=True)

if show_raw:
    st.dataframe(df_country.tail(10))


# Cached LSTM Model
@st.cache_resource
def train_model(series):
    """Trains a lightweight LSTM model once and reuses it for all reruns."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(np.array(series).reshape(-1, 1))

    look_back = 12
    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i + look_back])
        y.append(scaled[i + look_back])
    X, y = np.array(X), np.array(y)

    model = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, input_shape=(look_back, 1)),
        tf.keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=5, batch_size=8, verbose=0)
    return model, scaler

# Prepare data
df_series = df_country[['dt', 'SmoothedTemp']].copy()
df_series['Month'] = df_series['dt'].dt.to_period('M')
df_monthly = df_series.groupby('Month')['SmoothedTemp'].mean().reset_index()
df_monthly['Month'] = df_monthly['Month'].dt.to_timestamp()

# Train (cached)
model, scaler = train_model(df_monthly['SmoothedTemp'])

# AI Forecast
scaled = scaler.transform(np.array(df_monthly['SmoothedTemp']).reshape(-1, 1))
look_back = 12
last_seq = scaled[-look_back:]
future_preds = []

for _ in range(future_months):
    pred = model.predict(last_seq.reshape(1, look_back, 1), verbose=0)
    future_preds.append(pred[0, 0])
    last_seq = np.append(last_seq[1:], pred).reshape(look_back, 1)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))
future_dates = pd.date_range(df_monthly['Month'].iloc[-1] + pd.offsets.MonthBegin(),
                             periods=future_months, freq='MS')
df_future = pd.DataFrame({'Month': future_dates, 'PredictedTemperature': future_preds.flatten()})

# Plot Forecast
fig2 = go.Figure()
fig2.add_trace(go.Scatter(
    x=df_monthly['Month'], y=df_monthly['SmoothedTemp'],
    mode='lines', name='Historical (Smoothed)',
    line=dict(color='deepskyblue', width=2)
))
fig2.add_trace(go.Scatter(
    x=df_future['Month'], y=df_future['PredictedTemperature'],
    mode='lines', name='AI Forecast',
    line=dict(color='orange', width=3, dash='dash')
))
fig2.add_vrect(
    x0=df_future['Month'].iloc[0],
    x1=df_future['Month'].iloc[-1],
    fillcolor="orange", opacity=0.1, line_width=0
)
fig2.update_layout(
    title=f"AI Forecast for Next {future_months} Months in {country_filter}",
    xaxis_title="Date", yaxis_title="Temperature (°C)",
    template='plotly_dark', showlegend=True
)
st.plotly_chart(fig2, use_container_width=True)

# Summary

latest_temp = round(df_monthly['SmoothedTemp'].iloc[-1], 2)
forecast_end = round(df_future['PredictedTemperature'].iloc[-1], 2)
temp_change = forecast_end - latest_temp

st.markdown("### 📊 Forecast Summary")
col1, col2, col3 = st.columns(3)
col1.metric("Current Avg Temp", f"{latest_temp}°C")
col2.metric(f"Forecast ({future_months} mo)", f"{forecast_end}°C")
col3.metric("Change", f"{temp_change:+.2f}°C")

st.success(
    f"EcoAI predicts that the average temperature in **{country_filter}** "
    f"may change by **{temp_change:+.2f}°C** over the next **{future_months} months**."
)

# Disclaimer
st.warning("""
⚠️ **Disclaimer:**  
The forecast shown here is a **mathematical estimation** based on historical patterns.  
It does **not account for external factors** such as greenhouse gas emissions, 
policy changes, or natural variability.  
For verified projections, please consult **NASA**, **NOAA**, or **IPCC** data sources.
""")

# Cleanup
gc.collect()
