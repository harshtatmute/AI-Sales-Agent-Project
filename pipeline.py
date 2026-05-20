import joblib
import pandas as pd
from phi.agent import Agent
from phi.model.ollama import Ollama

print("🔄 Loading Machine Learning model and scaler artifacts...")

# 1. Load the saved ML components from disk
model = joblib.load('lead_scoring_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# 2. Initialize our ultra-lightweight Qwen Agent
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

print("✅ Systems initialized. Simulating a real customer input...")

# 3. Simulate a real customer inputting data on our website
# (This represents the raw data before scaling and encoding)
raw_customer_data = {
    'TotalVisits': 4.0,
    'Total Time Spent on Website': 1200, # 20 minutes
    'Page Views Per Visit': 3.0,
    'Lead Origin_Landing Page Submission': True,
    'Lead Source_Google': True,
    'Current Occupation_Working Professional': True,
    'Specialization_Marketing Management': True
}

# Convert the dictionary to a single-row DataFrame
input_df = pd.DataFrame([raw_customer_data])

# 4. Feature Scaling (Match exactly how we trained the model)
numeric_cols = ['TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# 5. Handle any columns the model expects but aren't in our raw input (Optimized)
model_features = model.feature_names_in_
missing_cols = {col: False for col in model_features if col not in input_df.columns}

# Convert all missing columns at once using pd.DataFrame and concat
if missing_cols:
    missing_df = pd.DataFrame([missing_cols], index=input_df.index)
    input_df = pd.concat([input_df, missing_df], axis=1)

# Reorder columns to perfectly match the trained model structure
input_df = input_df[model_features]

# 6. Run the ML Prediction Engine
probability = model.predict_proba(input_df)[:, 1][0]
lead_score = int(probability * 100)

print(f"\n🎯 Machine Learning Prediction Complete!")
print(f"🔥 Calculated Lead Score: {lead_score}/100")

# 7. Agentic Decision: Only trigger the AI Writer if the lead score is > 80
if lead_score >= 80:
    print("\n🚀 Hot Lead detected! Triggering Qwen Agent to write custom email...\n")
    agent_prompt = f"""
    Write a sales email for a Working Professional interested in Marketing Management.
    They scored a {lead_score}/100 interest level and spent 1200 seconds analyzing our page.
    """
    sales_email_agent.print_response(agent_prompt)
else:
    print("\nℹ️ Cold Lead. Score is below threshold. Email generation skipped.")
