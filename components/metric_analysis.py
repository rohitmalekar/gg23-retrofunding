import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

def render_metric_analysis(metrics_df, allocation_df):
    """
    Render the metric analysis section with various visualizations and insights.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing project metrics
        allocation_df (pd.DataFrame): DataFrame containing allocation data
    """

    # Generate explanation text
    explanation_text = """
    **What is this visualization showing?**  
    These visualizations analyze how different project metrics relate to final funding allocations. The analysis includes three key components:
    1. A correlation analysis showing how strongly each metric relates to final allocations
    2. An interactive scatter plot allowing you to explore individual metric relationships
    3. A feature importance analysis showing which metrics were most influential in predicting allocations

    """
    
    # Generate takeaways text
    takeaways_text = """
    **Key Insights:**
    
    - **GMV Growth is the Most Influential Metric:** Both the feature importance and correlation analyses highlight `gmv_growth` as the strongest predictor of final funding allocations. Projects with higher GMV growth consistently received larger allocations, indicating that overall economic activity is a primary driver in funding decisions.
    
    - **Developer and Donor Retention Matter:** Metrics like `developer_retention_rate` and `donor_retention_rate` show high relative importance, suggesting that projects able to retain contributors and donors are more likely to secure greater funding. This underscores the value placed on sustained engagement within projects.
    
    - **Active Participation and Onboarding are Key:** `current_active_developers` and `contributor_onboarding_rate` are also highly ranked, reflecting the importance of both ongoing activity and the ability to attract new contributors.
    
    - **Network Contribution Score Shows Negative or Weak Correlation:** Interestingly, `network_contribution_score` has a weak or even negative correlation with final allocations, indicating that this metric may not align closely with current funding priorities or may require further refinement.
    """
    
    # Render all components
    st.markdown(explanation_text)
    st.info(takeaways_text)
    
    st.subheader("Metric Correlations with Final Allocations")
    
    st.markdown("""
    **How to read this chart:**  
    - Bars pointing left (positive values) show metrics that tend to increase with higher funding
    - Bars pointing right (negative values) show metrics that tend to decrease with higher funding
    - The longer the bar, the stronger the relationship
    - Values closer to zero (shorter bars) indicate little to no relationship with funding
    """)

    # Clean and prepare data
    metrics_clean = metrics_df.copy()
    
    
    # Merge metrics with allocations
    analysis_df = pd.merge(metrics_clean, allocation_df[['project', 'total']], on='project')
    
    # Calculate correlations
    numeric_cols = metrics_clean.select_dtypes(include=[np.number]).columns
    correlations = analysis_df[numeric_cols].corrwith(analysis_df['total'])
    correlations = correlations.sort_values(ascending=False)
    
    # Create correlation visualization
    correlation_fig = px.bar(
        correlations,
        title="Correlation of Metrics with Final Allocations",
        labels={'value': 'Correlation Coefficient', 'index': 'Metric'},
        color=correlations,
        color_continuous_scale='RdBu'
    )
    correlation_fig.update_layout(showlegend=False)

    
    st.plotly_chart(correlation_fig, use_container_width=True)
    
    st.subheader("Individual Metric Analysis")
    
    st.markdown("""
    **How to use this chart:**  
    - Select a metric from the dropdown above to analyze its relationship with funding
    - Each point represents a project's metric value (x-axis) and its corresponding allocation (y-axis)
    - Hover over points to view detailed project information
    - A positive correlation (upward trend) indicates that higher metric values correspond to larger allocations
    - A negative correlation (downward trend) suggests that lower metric values are associated with larger allocations
    - A weak or no correlation (scattered points) indicates minimal influence of this metric on funding decisions
    """)
    
    # Create scatter plot for selected metric
    selected_metric = st.selectbox(
        "Select a metric to analyze",
        options=numeric_cols
    )
    
    scatter_fig = px.scatter(
        analysis_df,
        x=selected_metric,
        y='total',
        hover_data=['project'],
        title=f"{selected_metric} vs Final Allocation",
        labels={'total': 'Final Allocation ($)', selected_metric: selected_metric}
    )

    st.plotly_chart(scatter_fig, use_container_width=True)
    
    st.subheader("Feature Importance Analysis")
    
    st.markdown("""
    **What is Feature Importance?**  
    This analysis shows how much each metric contributes to predicting funding allocations when all metrics are considered together. Unlike the correlation analysis above which looks at individual relationships, this shows the relative influence of each metric in a combined model. Higher importance scores indicate metrics that have a stronger impact on funding decisions when working in conjunction with other metrics.
    """)
    
    # Feature Importance Analysis
    # Prepare data for regression
    X = analysis_df[numeric_cols]
    y = analysis_df['total']
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit regression model
    model = LinearRegression()
    model.fit(X_scaled, y)
    
    # Calculate feature importance
    importance = pd.DataFrame({
        'Metric': numeric_cols,
        'Importance': np.abs(model.coef_)
    })
    importance = importance.sort_values('Importance', ascending=False)
    
    # Plot feature importance
    importance_fig = px.bar(
        importance,
        x='Metric',
        y='Importance',
        title="Relative Importance of Metrics in Predicting Allocations",
        labels={'Importance': 'Relative Importance Score'}
    )
    
  
    
    st.plotly_chart(importance_fig, use_container_width=True) 

    retrospective_notes = """
    ---
    **Retrospective Notes (DRAFT)**
    
    To maximize the impact and fairness of future retro funding rounds, evaluation frameworks should balance quantitative metrics with qualitative context. While data-driven indicators like economic growth and engagement are powerful predictors, important qualitative factors—such as project quality, innovation, and community value—may not be fully captured by numbers alone. These can be brought into scope by enabling badgeholders to adjust allocations, incorporating peer reviews, or including narrative impact statements, ensuring a more holistic and equitable funding process.
    """
    
    st.markdown(retrospective_notes)