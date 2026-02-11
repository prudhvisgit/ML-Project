import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np

# Set Page Config
st.set_page_config(page_title="AeroGuard Pro: Engine Health", layout="wide")

# Load AI Model Assets
model = joblib.load('engine_model.pkl')
scaler = joblib.load('scaler.pkl')
features = joblib.load('feature_list.pkl')
df_sample = pd.read_csv('engine_data_cleaned.csv')

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("🚀 AeroGuard V2.0")
page = st.sidebar.radio("Navigate", ["Live Simulator", "Model Performance"])

# --- PAGE 1: LIVE SIMULATOR ---
if page == "Live Simulator":
    st.sidebar.header("🔧 Engine Simulation Controls")
    
    if st.sidebar.button("♻️ Reset to Baseline"):
        st.rerun()

    inputs = {}
    for feat in features:
        min_v, max_v = float(df_sample[feat].min()), float(df_sample[feat].max())
        mean_v = float(df_sample[feat].mean())
        inputs[feat] = st.sidebar.slider(f"{feat}", min_v, max_v, mean_v, key=f"s_{feat}")

    st.title("✈️ Engine Predictive Maintenance Dashboard")
    
    # Prediction Logic
    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df)
    prediction = int(model.predict(input_scaled)[0])

    # Key Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Predicted RUL", f"{prediction} Cycles", delta=f"{prediction - 100} vs Baseline")
    with col2:
        if prediction > 70:
            st.success("Status: HEALTHY")
            status = "HEALTHY"
        elif prediction > 30:
            st.warning("Status: WARNING")
            status = "WARNING"
        else:
            st.error("Status: DANGER")
            status = "DANGER"
    with col3:
        st.write(f"**Sensors Monitored:** {len(features)}")

    # Visualization
    st.subheader("📊 Current Sensor Profile vs. Average")
    avg_values = df_sample[features].mean()
    current_values = pd.Series(inputs)
    
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(features))
    ax.bar(x, avg_values, width=0.4, label='Dataset Average', color='gray', alpha=0.5)
    ax.bar(x + 0.4, current_values, width=0.4, label='Current Inputs', color='skyblue')
    ax.set_xticks(x + 0.2)
    ax.set_xticklabels(features, rotation=45)
    ax.legend()
    st.pyplot(fig)

    # Download Report
    report_data = input_df.copy()
    report_data['Predicted_RUL'] = prediction
    report_data['Health_Status'] = status
    csv = report_data.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Simulation Report (.csv)", csv, "report.csv", "text/csv")

# --- PAGE 2: MODEL PERFORMANCE ---
else:
    st.title("📊 Model Analytics & Validation")
    st.markdown("### How the AI perceives Engine Health")

    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💡 Feature Importance")
        st.write("Which sensors drive the prediction?")
        # Get importance from Random Forest
        importances = model.feature_importances_
        feat_imp = pd.Series(importances, index=features).nlargest(10)
        
        fig2, ax2 = plt.subplots()
        feat_imp.sort_values().plot(kind='barh', color='#ff4b4b', ax=ax2)
        ax2.set_xlabel("Impact Score")
        st.pyplot(fig2)
        
        

    with col2:
        st.subheader("📈 Degradation Curve")
        st.write("Typical RUL decay over time")
        # Visualizing a sample decay
        decay_data = pd.DataFrame({
            'Cycle': range(1, 101),
            'RUL Prediction': sorted(np.random.randint(20, 150, 100), reverse=True)
        })
        st.line_chart(decay_data.set_index('Cycle'))
        
        

    st.divider()
    st.info("💡 **Insights:** Sensors s11 and s12 are often the leading indicators of thermal stress in the High Pressure Compressor.")