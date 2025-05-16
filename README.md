# Gitcoin Retrospective Analysis Dashboard

This Streamlit application provides an interactive dashboard for analyzing Gitcoin's retroactive funding allocation data.

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