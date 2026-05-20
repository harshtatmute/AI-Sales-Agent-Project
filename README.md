# AI-Sales-Agent-Project

An end-to-end predictive and generative AI application that analyzes incoming sales leads, calculates conversion probabilities using a localized Machine Learning engine, and deploys an autonomous AI Agent to draft hyper-personalized sales copy for high-value targets.

---

## 🏗️ System Architecture

The application is engineered as a unified data pipeline split into three distinct modular layers:

[Raw User Input UI] ──► [Data Engineering Layer] (Scaling & 183-Feature Padding)
│
▼
[Logistic Regression Engine] ──► Calculates Probability %
│
(If Lead Score >= 80)
│
▼
[Qwen 2.5 AI Agent]
│
▼
[Personalized Sales Email]

1. **Data Engineering Layer:** Ingests raw customer features, executes missing data imputation patterns, encodes categorical values into binary markers, and scales numerical values.
2. **Predictive Machine Learning Engine:** Evaluates the processed feature space using a trained Logistic Regression model to compute a real-time conversion probability (0-100 Lead Score).
3. **Agentic AI Layer:** A conditional orchestration gate that activates an autonomous copywriter agent powered by **Phidata** and a local **Qwen 2.5** model to generate hyper-tailored email sequences for hot leads (scores ≥ 80).

---

## 📊 Performance & Key Metrics

* **Classification Accuracy:** **94.56%** on unseen testing data.
* **Feature Matrix Space:** Expanded from 30 messy columns to a clean **183-feature numeric matrix** via One-Hot Encoding.
* **Local Compute Footprint:** Leveraged an ultra-lightweight **350MB Qwen 2.5 (0.5B)** model to bypass local hardware RAM bottlenecks, optimizing execution speed to **~34 seconds per email generation** without data leaving local infrastructure.

---

## 🛠️ Tech Stack & Libraries

* **Core Language:** Python 3.13
* **Data Engineering & ML:** Pandas, Scikit-Learn, Joblib
* **Agentic Framework:** Phidata
* **Local LLM Engine:** Ollama (Qwen 2.5)
* **User Interface:** Streamlit (Full-Stack Web App)

---
