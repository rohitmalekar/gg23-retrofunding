# Gitcoin Retrospective Analysis Dashboard

This Streamlit application provides an interactive dashboard for analyzing Gitcoin's retroactive funding allocation data.

View application: https://gg23-retro-analysis.streamlit.app

**Badgeholder Preference Analysis - How badgeholders distributed their weights across metrics?**
<img width="1585" alt="Screenshot 2025-07-08 at 2 48 59 PM" src="https://github.com/user-attachments/assets/d772506a-2943-437c-bf71-2334ab00c6f0" />

**Metric Analysis - How do different project metrics relate to final funding allocations?**
<img width="1565" alt="Screenshot 2025-07-08 at 2 49 36 PM" src="https://github.com/user-attachments/assets/76a21e06-339c-4796-a3cc-1199be5dd312" />

**Performance Clusters Analysis - Were funds equitably deployed across clusters having different strengths and profiles?**
<img width="1575" alt="Screenshot 2025-07-08 at 2 50 02 PM" src="https://github.com/user-attachments/assets/deeef999-e51a-4313-bae9-0bc7e93d34ae" />




## Setup

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

Run the Streamlit app with:
```bash
streamlit run app.py
```

The application will open in your default web browser. If it doesn't, you can access it at http://localhost:8501

## Data Structure

The application uses three main datasets:
- `votes.csv`: Contains badgeholder votes for metrics
- `allocation.csv`: Contains final allocation results
- `metrics.csv`: Contains project metrics

These files should be located in the `retrospective/data/` directory. 
