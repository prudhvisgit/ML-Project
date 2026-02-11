# ML-Project 

# ✈️ Aero-Engine Predictive Maintenance Dashboard

### **Real-time Remaining Useful Life (RUL) Estimation using Machine Learning**

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-orange.svg)

## 📌 Project Overview
This project addresses a critical challenge in aviation: **predictive maintenance**. Using the NASA C-MAPSS dataset, I developed a machine learning pipeline that predicts the **Remaining Useful Life (RUL)** of a turbofan engine based on telemetry data from sensors measuring temperature, pressure, and fan speeds.



## 🚀 Key Features
* **Interactive Simulation:** Adjust engine parameters via sidebar sliders to see real-time RUL updates.
* **Intelligent Reset:** Quickly snap back to "Healthy" baseline values for sensitivity analysis.
* **Visual Analytics:** A dynamic bar chart compares your current simulation against the dataset's "Average Healthy Engine."
* **Automated Reporting:** Generate and download a `.csv` report of your simulation results.

## 📊 Model Performance
The application uses a **Random Forest Regressor** trained on historical degradation trajectories.
* **Mean Absolute Error (MAE):** ~29.7 Cycles
* **Key Indicators:** The model identifies **s11**, **s12**, and **s4** as the primary drivers of engine failure.



## 💻 How to Run Locally

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/yourusername/engine-predictive-maintenance.git](https://github.com/yourusername/engine-predictive-maintenance.git)
   cd engine-predictive-maintenance
