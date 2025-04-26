import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import base64

# Load the trained model
with open("kidney_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load the graph data
with open("kindney_graph.pkl", "rb") as f:
    graph_data = pickle.load(f)

# Set background function (using local image converted to base64)
def set_bg():
    with open("Opera Snapshot_2025-04-26_180435_pngtree.com.png", "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    
    page_bg_img = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    /* Text Styling */
    h1, h2, h3, h4, h5, h6, p, label, span {{
        color: #222222 !important; /* Dark text */
        font-weight: 600;
    }}
    /* Input box styling */
    input, textarea {{
        background: rgba(255, 255, 255, 0.85) !important;
        color: #222222 !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.6em;
        border: 1px solid #ccc;
    }}
    /* Button styling */
    div.stButton > button:first-child {{
        background-color: #0066cc;
        color: white;
        font-weight: bold;
        height: 3em;
        width: 16em;
        border-radius: 8px;
        border: none;
        font-size: 18px;
        transition: background-color 0.3s ease;
    }}
    div.stButton > button:first-child:hover {{
        background-color: #004999;
        color: white;
    }}
    </style>
    """
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set background
set_bg()

# Title and instructions
st.markdown("<h1 style='text-align:center;'>Kidney Disease Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; font-size:18px;'>Enter the parameters below to check your kidney health status.</p>", unsafe_allow_html=True)

# Section title
st.subheader("📝 Enter the 10 Parameters:")

# List of features
feature_names = [
    'Age', 'Blood Pressure (bp)', 'Specific Gravity (sg)', 'Albumin (al)', 'Sugar (su)', 
    'Blood Glucose Random (bgr)', 'Blood Urea (bu)', 'Serum Creatinine (sc)', 
    'Hemoglobin (hemo)', 'Packed Cell Volume (pcv)'
]

# Collect user inputs
user_inputs = []
for i, feature in enumerate(feature_names):
    value = st.text_input(f"{feature}:", key=f"input_{i}")
    user_inputs.append(value)

# Predict button
if st.button("🔎 Predict"):
    try:
        numeric_inputs = np.array(user_inputs, dtype=float).reshape(1, -1)
        prediction = model.predict(numeric_inputs)[0]
        
        if prediction == 1:
            st.error("🚨 You have a chronic kidney disease!")
        else:
            st.success("✅ No disease detected. You are good!")
    except ValueError:
        st.warning("⚠ Please enter valid numerical inputs for all fields.")

# Graph section
st.subheader("📊 Actual vs. Predicted Results")
fig, ax = plt.subplots(figsize=(8, 4))
graph_data.plot(kind='bar', ax=ax, color=["#3399ff", "#ff6666"])  # cool blue and soft red
ax.set_xlabel("Samples", fontsize=12, fontweight='bold')
ax.set_ylabel("Values", fontsize=12, fontweight='bold')
ax.set_title("Prediction Comparison", fontsize=14, fontweight='bold')
st.pyplot(fig)
