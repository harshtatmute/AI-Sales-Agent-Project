import streamlit as st
import joblib
import pandas as pd
from phi.agent import Agent
from phi.model.ollama import Ollama

# Set up page configurations
st.set_page_config(page_title="AI Lead Scoring & Sales Agent", layout="wide")

st.title("🎯 AI-Powered Lead Scoring & Agentic Sales Assistant")
st.write("Input user details below to predict conversion probability and generate personalized sales emails.")

# 1. Load the backend ML models
@st.cache_resource
def load_resources():
    model = joblib.load('lead_scoring_model.pkl')
    scaler = joblib.load('feature_scaler.pkl')
    return model, scaler

try:
    model, scaler = load_resources()
except Exception as e:
    st.error(f"Error loading model files: {e}")

# 2. Create Layout Columns for Input Form
col1, col2 = st.columns(2)

with col1:
    st.header("👤 Lead Profile Data")
    
    # Numerical Inputs
    total_visits = st.number_input("Total Website Visits", min_value=0.0, max_value=100.0, value=4.0, step=1.0)
    time_spent = st.slider("Total Time Spent on Website (Seconds)", min_value=0, max_value=3000, value=1200, step=10)
    page_views = st.number_input("Page Views Per Visit", min_value=0.0, max_value=50.0, value=3.0, step=0.5)
    
    # Categorical Selection Inputs
    occupation = st.selectbox("Current Occupation", ["Working Professional", "Student", "Unemployed", "Other"])
    specialization = st.selectbox("Area of Specialization", ["Marketing Management", "Digital Marketing", "Business Administration", "Finance", "Other"])
    lead_source = st.selectbox("Lead Source", ["Google", "Direct Traffic", "Organic Search", "Reference", "Other"])
    lead_origin = st.selectbox("Lead Origin", ["Landing Page Submission", "API", "Lead Add Form", "Other"])

# 3. Process inputs upon button click
with col2:
    st.header("🤖 Model Predictions & AI Output")
    
    if st.button("Calculate Lead Score & Draft Email", type="primary"):
        # Map structural dictionary
        raw_input = {
            'TotalVisits': total_visits,
            'Total Time Spent on Website': time_spent,
            'Page Views Per Visit': page_views,
            f'Current Occupation_{occupation}': True,
            f'Specialization_{specialization}': True,
            f'Lead Source_{lead_source}': True,
            f'Lead Origin_{lead_origin}': True
        }
        
        input_df = pd.DataFrame([raw_input])
        
        # Continuous Feature Scaling
        numeric_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
        input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
        
        # Align structural columns with full feature space
        model_features = model.feature_names_in_
        missing_cols = {col: False for col in model_features if col not in input_df.columns}
        
        if missing_cols:
            missing_df = pd.DataFrame([missing_cols], index=input_df.index)
            input_df = pd.concat([input_df, missing_df], axis=1)
            
        input_df = input_df[model_features]
        
        # Model Evaluation Execution
        probability = model.predict_proba(input_df)[:, 1][0]
        lead_score = int(probability * 100)
        
        # Metrics UI Visualization display
        st.metric(label="Calculated Lead Score", value=f"{lead_score} / 100")
        
        if lead_score >= 80:
            st.success("🔥 High-Converting Hot Lead Detected!")
            st.subheader("📝 Automated Personalized Sales Outreach:")
            
            # Initialize local AI writer agent
            sales_email_agent = Agent(
                name="Sales Personalization Agent",
                model=Ollama(id="qwen2.5:0.5b"),
                description="You are an expert sales copywriter at X Education.",
                instructions=[
                    "Write a targeted, warm outreach email based on the lead profile.",
                    "Mention how our specialized courses solve their career problems.",
                    "Sign off as 'The X Education Team'. Do not use placeholder brackets."
                ],
                markdown=True,
            )
            
            agent_prompt = f"Write a professional sales email for a {occupation} interested in {specialization}. They scored a {lead_score}/100 interest level and spent {time_spent} seconds analyzing our page."
            
            # Run and capture agent stream directly to the web container interface
            with st.spinner("AI Agent is crafting your email..."):
                response_placeholder = st.empty()
                # Capture standard print output from phidata stream using a clean UI block
                response = sales_email_agent.run(agent_prompt)
                response_placeholder.markdown(response.content)
        else:
            st.info("ℹ️ Cold Lead. Score is below operational engagement thresholds (80/100). Automated outreach paused.")