from phi.agent import Agent
from phi.model.ollama import Ollama

# 1. Initialize our localized, ultra-lightweight Qwen Agent
sales_email_agent = Agent(
    name="Sales Personalization Agent",
    model=Ollama(id="qwen2.5:0.5b"),  # Using the ultra-lightweight 350MB model to save RAM
    description="You are an expert sales copywriter at X Education.",
    instructions=[
        "Analyze the lead's profile data (Occupation, Time Spent, Specialization).",
        "Write a highly targeted, warm outreach email matching their current professional status.",
        "Mention how the course can help them solve specific career problems based on their profile.",
        "Keep it concise, professional, and end with a call-to-action to book a counseling session.",
        "Do not use generic bracket placeholders like [Your Name] or [Company Name]. Sign off as 'The X Education Team'."
    ],
    markdown=True,
)

# 2. Mock data for a high-scoring 'Hot Lead' from our ML pipeline
sample_hot_lead = {
    "Lead_Score": 96,
    "Occupation": "Working Professional",
    "Specialization": "Digital Marketing",
    "Total_Time_Spent_Seconds": 1532,
    "Page_Views_Per_Visit": 4.5,
    "City": "Mumbai"
}

# 3. Create the prompt injecting the data dynamically
prompt = f"""
Please generate a personalized sales email for the following lead:
- Lead Score: {sample_hot_lead['Lead_Score']}/100 (Extremely High Interest)
- Current Occupation: {sample_hot_lead['Occupation']}
- Specialization of Interest: {sample_hot_lead['Specialization']}
- Time Spent exploring our courses: {sample_hot_lead['Total_Time_Spent_Seconds']} seconds
- Location: {sample_hot_lead['City']}

Tailor the pitch directly to a {sample_hot_lead['Occupation']} from {sample_hot_lead['City']} interested in {sample_hot_lead['Specialization']}.
"""

print("Sending hot lead profile to Qwen ultra-light Agent...\n")

# 4. Run the agent and print the response directly to the console
sales_email_agent.print_response(prompt)