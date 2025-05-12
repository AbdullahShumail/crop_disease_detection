import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import json

# Page configuration
st.set_page_config(
    page_title="Soybean Disease Predictor",
    page_icon="🌱",
    layout="wide"
)

# Initialize session state for history
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []

# Load saved models and preprocessors
@st.cache_resource
def load_models():
    model = joblib.load('soybean_models/soybean_model.pkl')
    scaler = joblib.load('soybean_models/scaler.pkl')
    label_encoders = joblib.load('soybean_models/label_encoders.pkl')
    feature_names = joblib.load('soybean_models/feature_names.pkl')
    classes = joblib.load('soybean_models/classes.pkl')
    return model, scaler, label_encoders, feature_names, classes

# Load sample data for reference
@st.cache_data
def load_sample_data():
    return pd.read_csv('soybean_models/sample_data.csv')

model, scaler, label_encoders, feature_names, classes = load_models()
sample_data = load_sample_data()

# Title and description
st.title("🌱 Soybean Disease Prediction System")
st.markdown("""
This system predicts soybean plant diseases based on various crop features and symptoms.
Fill in the features below to get a disease prediction.
""")

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Prediction", "History", "Data Reference", "About"])

with tab1:
    st.header("Disease Prediction")
    
    # Create form for input
    with st.form("prediction_form"):
        st.subheader("Enter Crop Features")
        
        # Create columns for better layout
        col1, col2, col3 = st.columns(3)
        
        # Dictionary to store input values
        input_features = {}
        
        # Organize features into categories for better UX
        categorical_features = [col for col in feature_names if col in label_encoders]
        numerical_features = [col for col in feature_names if col not in label_encoders]
        
        # Display categorical features
        with col1:
            st.markdown("### Environmental Factors")
            for i, feature in enumerate(categorical_features[:len(categorical_features)//3]):
                if feature in sample_data.columns:
                    unique_values = sample_data[feature].dropna().unique()
                    input_features[feature] = st.selectbox(
                        feature.replace('-', ' ').title(),
                        options=unique_values,
                        key=f"cat_{feature}"
                    )
        
        with col2:
            st.markdown("### Plant Symptoms")
            for i, feature in enumerate(categorical_features[len(categorical_features)//3:2*len(categorical_features)//3]):
                if feature in sample_data.columns:
                    unique_values = sample_data[feature].dropna().unique()
                    input_features[feature] = st.selectbox(
                        feature.replace('-', ' ').title(),
                        options=unique_values,
                        key=f"cat_{feature}"
                    )
        
        with col3:
            st.markdown("### Additional Features")
            for i, feature in enumerate(categorical_features[2*len(categorical_features)//3:]):
                if feature in sample_data.columns:
                    unique_values = sample_data[feature].dropna().unique()
                    input_features[feature] = st.selectbox(
                        feature.replace('-', ' ').title(),
                        options=unique_values,
                        key=f"cat_{feature}"
                    )
        
        # Numerical features (if any)
        if numerical_features:
            st.markdown("### Numerical Measurements")
            num_cols = st.columns(3)
            for i, feature in enumerate(numerical_features):
                with num_cols[i % 3]:
                    input_features[feature] = st.number_input(
                        feature.replace('-', ' ').title(),
                        value=0.0,
                        key=f"num_{feature}"
                    )
        
        # Prediction button
        submitted = st.form_submit_button("Predict Disease", type="primary")
    
    if submitted:
        # Prepare features for prediction
        features_df = pd.DataFrame([input_features])
        
        # Apply label encoding
        for col in features_df.columns:
            if col in label_encoders:
                le = label_encoders[col]
                value = features_df[col].iloc[0]
                if value in le.classes_:
                    features_df[col] = le.transform([value])[0]
                else:
                    features_df[col] = 0
        
        # Ensure all columns are present
        for col in feature_names:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns
        features_df = features_df[feature_names]
        
        # Scale features
        features_scaled = scaler.transform(features_df.values)
        
        # Make prediction
        prediction = model.predict(features_scaled)[0]
        probabilities = model.predict_proba(features_scaled)[0]
        
        # Save to history
        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'prediction': prediction,
            'confidence': max(probabilities),
            'features': input_features,
            'all_probabilities': {cls: float(prob) for cls, prob in zip(classes, probabilities)}
        }
        st.session_state.prediction_history.append(history_entry)
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create columns for results
        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.metric(
                label="Predicted Disease",
                value=prediction,
                delta=None
            )
        
        with result_col2:
            confidence = max(probabilities) * 100
            st.metric(
                label="Confidence",
                value=f"{confidence:.1f}%",
                delta=None
            )
        
        with result_col3:
            # Risk level based on disease type
            if prediction in ['healthy', 'normal']:
                risk_level = "Low"
                risk_color = "green"
            elif prediction in ['bacterial', 'virus']:
                risk_level = "High"
                risk_color = "red"
            else:
                risk_level = "Medium"
                risk_color = "orange"
            
            st.metric(
                label="Risk Level",
                value=risk_level,
                delta=None
            )
        
        # Probability distribution chart
        st.markdown("### Probability Distribution")
        
        prob_df = pd.DataFrame({
            'Disease': classes,
            'Probability': probabilities * 100
        }).sort_values('Probability', ascending=False)
        
        fig = px.bar(
            prob_df,
            x='Disease',
            y='Probability',
            title="Disease Probability Distribution",
            color='Probability',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(
            yaxis_title="Probability (%)",
            xaxis_tickangle=-45,
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Recommendations based on prediction
        st.markdown("### Recommendations")
        
        recommendations = {
            'diaporthe-stem-canker': """
            ⚠️ **Diaporthe Stem Canker Detected**
            - Use resistant varieties
            - Practice crop rotation
            - Apply fungicides at appropriate times
            - Destroy infected plant debris
            """,
            'charcoal-rot': """
            🚨 **Charcoal Rot Detected**
            - Improve soil drainage
            - Avoid drought stress
            - Use resistant varieties
            - Consider crop rotation with non-host crops
            """,
            'rhizoctonia-root-rot': """
            ⚠️ **Rhizoctonia Root Rot Detected**
            - Improve soil drainage
            - Avoid deep planting
            - Use fungicide seed treatments
            - Practice crop rotation
            """,
            'phytophthora-rot': """
            🚨 **Phytophthora Rot Detected**
            - Improve field drainage
            - Use resistant varieties
            - Apply fungicides
            - Avoid working in wet fields
            """,
            'bacterial-blight': """
            ⚠️ **Bacterial Blight Detected**
            - Use disease-free seed
            - Apply copper-based bactericides
            - Remove infected plant debris
            - Practice crop rotation
            """,
            'bacterial-pustule': """
            ⚠️ **Bacterial Pustule Detected**
            - Use resistant varieties
            - Apply copper-based treatments
            - Practice good field sanitation
            - Avoid working in wet fields
            """,
            'purple-seed-stain': """
            ⚠️ **Purple Seed Stain Detected**
            - Use fungicide treatments
            - Plant high-quality seed
            - Harvest at appropriate moisture
            - Store seeds properly
            """
        }
        
        if prediction.lower() in ['healthy', 'normal']:
            st.success("✅ Your soybean crop appears to be healthy! Continue with regular monitoring.")
        else:
            if prediction in recommendations:
                st.warning(recommendations[prediction])
            else:
                st.info(f"Disease detected: {prediction}. Consult with an agricultural expert for specific treatment recommendations.")

with tab2:
    st.header("Prediction History")
    
    if st.session_state.prediction_history:
        # Create DataFrame from history
        history_df = pd.DataFrame([
            {
                'Timestamp': entry['timestamp'],
                'Prediction': entry['prediction'],
                'Confidence': f"{entry['confidence']*100:.1f}%"
            }
            for entry in st.session_state.prediction_history
        ])
        
        # Display history table
        st.dataframe(history_df, use_container_width=True)
        
        # Plot history trends
        if len(history_df) > 1:
            st.subheader("Prediction Trends")
            
            # Disease frequency
            disease_counts = history_df['Prediction'].value_counts()
            
            fig = px.pie(
                values=disease_counts.values,
                names=disease_counts.index,
                title="Disease Distribution in History"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Download history
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Clear History"):
                st.session_state.prediction_history = []
                st.rerun()
        
        with col2:
            # Export history as JSON
            history_json = json.dumps(st.session_state.prediction_history, indent=2)
            st.download_button(
                label="Download History (JSON)",
                data=history_json,
                file_name=f"soybean_prediction_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    else:
        st.info("No predictions made yet. Go to the Prediction tab to start.")

with tab3:
    st.header("Data Reference")
    st.markdown("### Sample Data")
    st.dataframe(sample_data, use_container_width=True)
    
    st.markdown("### Feature Descriptions")
    st.markdown("""
    Here are the features used for prediction:
    
    - **Environmental Factors**: date, plant-stand, precip, temp, hail, crop-hist, area-damaged
    - **Plant Symptoms**: leaves, stem, lodging, seed-tmt, germination
    - **Disease Indicators**: leaf-halo, leaf-marg, leaf-size, leaf-shread, leaf-malf, leaf-mild
    - **Additional Features**: stem-cankers, canker-lesion, fruiting-bodies, external-decay, etc.
    
    Each feature helps the model identify specific disease patterns in soybean crops.
    """)

with tab4:
    st.header("About")
    st.markdown("""
    ### Soybean Disease Prediction System
    
    This system uses machine learning to predict soybean diseases based on crop features and environmental conditions.
    
    **Model Information:**
    - Algorithm: Random Forest Classifier
    - Features: Multiple categorical and numerical features
    - Diseases: Multiple soybean diseases including stem canker, root rot, bacterial blight, etc.
    
    **How to Use:**
    1. Go to the Prediction tab
    2. Fill in the crop features based on your observations
    3. Click "Predict Disease" to get results
    4. View recommendations for treatment
    5. Check the History tab to see past predictions
    
    **Disclaimer:**
    This is a prediction tool based on machine learning. Always consult with agricultural experts for critical decisions.
    """)

# Sidebar
with st.sidebar:
    st.header("System Status")
    st.success("Model Loaded")
    st.info(f"Total Predictions: {len(st.session_state.prediction_history)}")
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    if st.session_state.prediction_history:
        recent_predictions = [entry['prediction'] for entry in st.session_state.prediction_history[-10:]]
        most_common = max(set(recent_predictions), key=recent_predictions.count)
        st.metric("Most Common Disease (Recent)", most_common)
    
    st.markdown("---")
    st.markdown("### Resources")
    st.markdown("""
    - [Soybean Disease Guide](https://www.extension.iastate.edu)
    - [Crop Protection Network](https://www.cropprotectionnetwork.org)
    - [USDA Plant Disease Info](https://www.usda.gov)
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>🌱 Soybean Disease Prediction System - Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)