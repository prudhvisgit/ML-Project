import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Set Page Config
st.set_page_config(page_title="Aero-Engine Health Dashboard", layout="wide")

# Load AI Model Assets
model = joblib.load('engine_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_list.pkl')
df_sample = pd.read_csv('engine_data_cleaned.csv')

# --- SIDEBAR: CONTROLS ---
st.sidebar.header("🔧 Engine Simulation Controls")

if st.sidebar.button("♻️ Reset to Baseline"):
    st.rerun()

inputs = {}
for feat in features:
    min_v, max_v = float(df_sample[feat].min()), float(df_sample[feat].max())
    mean_v = float(df_sample[feat].mean())
    inputs[feat] = st.sidebar.slider(f"{feat}", min_v, max_v, mean_v, key=f"s_{feat}")

# --- MAIN UI ---
st.title("✈️ Engine Predictive Maintenance Dashboard")

# 1. Prediction Logic
input_df = pd.DataFrame([inputs])
input_scaled = scaler.transform(input_df)
prediction = int(model.predict(input_scaled)[0])

# 2. Key Metrics Display
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted RUL", f"{prediction} Cycles")
with col2:
    status = "HEALTHY" if prediction > 70 else "WARNING" if prediction > 30 else "DANGER"
    st.write(f"**Engine Status:** {status}")
with col3:
    st.write(f"**Sensors Monitored:** {len(features)}")

# 3. VISUALIZATION: Feature Comparison
st.subheader("📊 Current Sensor Profile vs. Average")
# Prepare data for plotting
avg_values = df_sample[features].mean()
current_values = pd.Series(inputs)

fig, ax = plt.subplots(figsize=(10, 4))
x = range(len(features))
ax.bar(x, avg_values, width=0.4, label='Dataset Average', align='center', color='gray', alpha=0.5)
ax.bar(x, current_values, width=0.4, label='Current Inputs', align='edge', color='skyblue')
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45)
ax.set_ylabel("Sensor Values")
ax.legend()
st.pyplot(fig)



# 4. DOWNLOAD REPORT
st.subheader("📂 Export Simulation Data")
report_data = input_df.copy()
report_data['Predicted_RUL'] = prediction
report_data['Health_Status'] = status

csv = report_data.to_csv(index=False).encode('utf-8')

st.download_button(
    label="📥 Download Simulation Report (.csv)",
    data=csv,
    file_name='engine_health_report.csv',
    mime='text/csv',
)