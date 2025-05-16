import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def calculate_variance(weights):
    """
    Calculate variance of weights for a badgeholder.
    
    Args:
        weights (pd.Series): Series of weights for a badgeholder
        
    Returns:
        float: Variance of weights
    """
    return np.var(weights)

def analyze_polarization(votes_df):
    """
    Analyze the variance in badgeholder preferences using boxplots.
    
    Args:
        votes_df (pd.DataFrame): DataFrame containing voting data
        
    Returns:
        tuple: (figure, expert_stats, regular_stats)
    """
    
    metric_columns = [col for col in votes_df.columns if col in ['contributor_onboarding_rate','current_active_developers','developer_retention_rate','donor_retention_rate','gmv_growth','network_contribution_score','scaling_community_rounds','unique_donor_growth']]
    
    # Calculate variance for each badgeholder
    votes_df['variance'] = votes_df[metric_columns].apply(calculate_variance, axis=1)
    
    # Create badgeholder type column
    votes_df['badgeholder_type'] = votes_df['Weight'].map({60: 'Expert', 20: 'Regular'})
    
    # Create boxplot
    fig = px.box(
        votes_df,
        x='badgeholder_type',
        y='variance',
        color='badgeholder_type',
        points='all',  # Show all points
        title='Variance in Metric Weights by Badgeholder Tier',
        labels={
            'variance': 'Variance in Weights',
            'badgeholder_type': 'Badgeholder Type'
        }
    )
    
    # Update layout
    fig.update_layout(
        height=600,
        showlegend=False,  # Hide legend since it's redundant
        yaxis_title="Variance in Weights",
        xaxis_title="Badgeholder Type"
    )
    
    # Calculate statistics by badgeholder type
    expert_stats = votes_df[votes_df['Weight'] == 60].describe()
    regular_stats = votes_df[votes_df['Weight'] == 20].describe()
    
    return fig, expert_stats, regular_stats

def analyze_polarization_per_metric(votes_df):
    """
    Analyze the variance in badgeholder preferences for each metric separately.
    
    Args:
        votes_df (pd.DataFrame): DataFrame containing voting data
        
    Returns:
        tuple: (figure, metric_stats)
    """
    
    # Define the expected metrics explicitly
    expected_metrics = [
        'contributor_onboarding_rate',
        'current_active_developers',
        'developer_retention_rate',
        'donor_retention_rate',
        'gmv_growth',
        'network_contribution_score',
        'scaling_community_rounds',
        'unique_donor_growth'
    ]
    
    # Get the metrics that exist in the DataFrame
    metric_columns = [col for col in expected_metrics if col in votes_df.columns]
    
    # Create badgeholder type column
    votes_df['badgeholder_type'] = votes_df['Weight'].map({60: 'Expert', 20: 'Regular'})
    
    # Calculate number of rows and columns for subplots
    n_metrics = len(metric_columns)
    n_cols = 2  
    n_rows = 4  
    
    # Create subplot figure
    fig = make_subplots(
        rows=n_rows, 
        cols=n_cols,
        subplot_titles=metric_columns,
        vertical_spacing=0.15,  # Increased spacing
        horizontal_spacing=0.1,
        specs=[[{"type": "box"}, {"type": "box"}] for _ in range(n_rows)]  # Explicitly specify box type for each subplot
    )
    
    # Create boxplot for each metric
    for idx, metric in enumerate(metric_columns):
        row = idx // n_cols + 1
        col = idx % n_cols + 1
        
        
        # Add boxplot for experts
        expert_data = votes_df[votes_df['badgeholder_type'] == 'Expert'][metric]
        fig.add_trace(
            go.Box(
                y=expert_data,
                name='Expert',
                marker_color='#1f77b4',
                boxpoints='all',
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
        
        # Add boxplot for regular badgeholders
        regular_data = votes_df[votes_df['badgeholder_type'] == 'Regular'][metric]
        fig.add_trace(
            go.Box(
                y=regular_data,
                name='Regular',
                marker_color='#ff7f0e',
                boxpoints='all',
                showlegend=(idx == 0)
            ),
            row=row, col=col
        )
    
    # Update layout
    fig.update_layout(
        height=1600,  # Increased height
        width=1000,
        title_text="Distribution of Weights by Metric and Badgeholder Type",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5
        )
    )
    
    # Update y-axis labels
    for i in range(1, n_rows + 1):
        for j in range(1, n_cols + 1):
            fig.update_yaxes(title_text="Weight", row=i, col=j)
    
    # Calculate statistics for each metric
    metric_stats = {}
    for metric in metric_columns:
        metric_stats[metric] = {
            'expert': votes_df[votes_df['Weight'] == 60][metric].describe(),
            'regular': votes_df[votes_df['Weight'] == 20][metric].describe()
        }
    
    return fig, metric_stats

def render_polarization_analysis(votes_df):
    """
    Perform polarization analysis on the voting data.
    
    Args:
        votes_df (pd.DataFrame): DataFrame containing voting data
        
    Returns:
        tuple: (figure, expert_stats, regular_stats, per_metric_fig, metric_stats, explanation_text, takeaways_text)
    """
    # Get overall analysis results
    fig, expert_stats, regular_stats = analyze_polarization(votes_df)
    
    # Get per-metric analysis results
    per_metric_fig, metric_stats = analyze_polarization_per_metric(votes_df)
    
    # Define explanation text
    explanation_text = """
    **What is this visualization showing?**  
    These visualizations show how badgeholders distributed their weights across metrics. The first plot shows the overall variance in weight distribution, while the subsequent plots break this down by individual metric.

    **How to read the plots:**
    - Each box shows the interquartile range (IQR) - the middle 50% of the data
    - The line in the middle of the box is the median
    - The whiskers extend to the most extreme points within 1.5 * IQR
    - Individual points show each badgeholder's weight for that metric
    - Higher variance means more uneven distribution of weights
    """
    
    # Define takeaways text
    takeaways_text = """            
    **Key Insights:**
    - Experts exhibit greater variability and less consensus in their metric weightings, with a wider spread and more outliers across nearly all metrics.
    - Regular badgeholders show more consistency, with weights tightly clustered for most metrics, indicating stronger agreement within this group.
    - Aggregate view: Experts tend to assign higher and more varied weights to metrics like contributor_onboarding_rate and gmv_growth, while Regulars generally favor retention-related metrics such as developer_retention_rate and donor_retention_rate with more uniform weightings.
    - For all metrics, Experts' votes are more dispersed, while Regulars' votes are more concentrated, reflecting a pattern of division versus alignment between the two cohorts.
    """
    
    # Retrospective Notes
    retrospective_notes = """
    ---
    **Retrospective Notes (DRAFT)**

    - **Engage Both Cohorts for Balanced Perspectives:**  
      Since Experts show more diverse and less consistent preferences, while Regular badgeholders are more aligned and consistent, it's important to include both groups in future voting rounds. This ensures a balance between innovative, varied viewpoints (from Experts) and stable, consensus-driven input (from Regulars).

    - **Clarify Metric Importance:**  
      Consider providing clearer guidance or discussion opportunities around the importance of each metric, especially for metrics where Expert opinions are highly dispersed. This may help reduce polarization and foster more informed, aligned decision-making.

    - **Monitor and Address Outliers:**  
      Pay attention to outlier votes, as they may indicate either innovative thinking or misunderstanding of the metrics. Follow-up discussions or feedback loops could help clarify intentions and improve future rounds.

    **Summary:**  
    A mixed approach—leveraging both the diversity of Experts and the consensus of Regulars—will likely yield the most robust and representative metric weighting outcomes in future rounds.
    """
    
    return fig, expert_stats, regular_stats, per_metric_fig, metric_stats, explanation_text, takeaways_text, retrospective_notes
