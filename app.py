import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from components.polarization_analysis import render_polarization_analysis
from components.metric_analysis import render_metric_analysis
from components.performance_clusters import render_performance_clusters

# Set page config
st.set_page_config(
    page_title="Gitcoin Retrospective Analysis",
    page_icon="📊",
    layout="wide"
)

# Title
st.title("GG23 Mature Builders Retro Funding Round - Retrospective Dashboard")

# Load data
@st.cache_data
def load_data():
    # Read votes data with explicit handling of newlines in headers
    votes_df = pd.read_csv('./data/votes.csv', lineterminator='\n')
    
    # Fix the donor_retention_rate column name
    votes_df.columns = [col.replace('\n', '') for col in votes_df.columns]
    
    # Load other data
    allocation_df = pd.read_csv('./data/allocation.csv')
    metrics_df = pd.read_csv('./data/metrics.csv')
    

    
    # Clean monetary columns in allocation_df
    for col in ['base', 'incremental', 'total']:
        allocation_df[col] = allocation_df[col].str.replace('$', '').str.replace(',', '').astype(float)
    
    return votes_df, allocation_df, metrics_df

try:
    votes_df, allocation_df, metrics_df = load_data()
    
    # Display basic information
    st.header("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Number of Projects", len(allocation_df))
    with col2:
        st.metric("Number of Voters", len(votes_df))
    with col3:
        st.metric("Total Allocation", f"${allocation_df['total'].sum():,.2f}")

    
    tab1, tab2, tab3, tab4 = st.tabs(["Badgeholder Preference Analysis", "Metric Analysis", "Performance Clusters", "Raw Data"])
    
    with tab1:
        # Badgeholder Preference Analysis Section
        st.header("Badgeholder Preference Analysis")
            
        # Get analysis results
        fig, expert_stats, regular_stats, per_metric_fig, metric_stats, explanation_text, takeaways_text, retrospective_notes = render_polarization_analysis(votes_df)
        
        # Display explanation and visualization
        st.markdown(explanation_text)
        st.info(takeaways_text)
        
        # Display overall variance analysis
        st.subheader("Overall Weight Distribution Variance")
        st.plotly_chart(fig, use_container_width=True)
        
        # Display per-metric analysis
        st.subheader("Weight Distribution by Metric")
        st.plotly_chart(per_metric_fig, use_container_width=True)

        st.markdown(retrospective_notes)
        
        
    with tab2:
        # Metric Analysis Section
        st.header("Metric Analysis")
        
        # Render metric analysis
        render_metric_analysis(metrics_df, allocation_df)

    with tab3:
        # Performance Clusters Section
        st.header("Performance Clusters Analysis")
        
        # Render performance cluster analysis
        render_performance_clusters(metrics_df, allocation_df)

    with tab4:
        tab1, tab2, tab3 = st.tabs(["Allocations", "Votes", "Metrics"])
        
        with tab1:
            st.dataframe(allocation_df)
        with tab2:
            st.dataframe(votes_df)
        with tab3:
            st.dataframe(metrics_df)

except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    st.info("Please ensure the data files are in the correct location") 