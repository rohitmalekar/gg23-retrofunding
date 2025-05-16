import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import plotly.express as px
import plotly.graph_objects as go

def render_performance_clusters(metrics_df, allocation_df):
    """
    Render performance cluster analysis visualization.
    
    Args:
        metrics_df (pd.DataFrame): DataFrame containing project metrics
        allocation_df (pd.DataFrame): DataFrame containing allocation information
    """
    # 1. Prepare the data
    # Select numeric columns for clustering
    numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Remove any columns that shouldn't be used for clustering
    exclude_cols = ['project']  # Add any other columns to exclude
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Create feature matrix
    X = metrics_df[numeric_cols].copy()
    
    # Handle missing values
    X = X.fillna(X.mean())
    
    # Normalize the data
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 2. Run K-Means clustering
    n_clusters = 4  # You can adjust this or make it dynamic
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    
    # 3. Reduce dimensions for visualization
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    
    # Create visualization dataframe
    viz_df = pd.DataFrame({
        'PC1': X_pca[:, 0],
        'PC2': X_pca[:, 1],
        'Cluster': clusters,
        'Project': metrics_df['project'],
        'Grant Amount': allocation_df['total']
    })
    
    # Calculate explained variance ratio
    explained_variance = pca.explained_variance_ratio_
    
    # Add user-friendly description
    st.markdown("""
    ### How to Read This Visualization
    
    This scatter plot shows how projects are grouped based on their performance metrics:
    
    - **Each point** represents a project
    - **Point size** indicates the grant amount (larger points = larger grants)
    - **Point color** shows which cluster the project belongs to:
        - 🔴 Red: Stable, steadily growing projects
        - 🔵 Blue: Projects with loyal donor base
        - 🟢 Green: High-growth, breakout projects
        - 🟣 Purple: Early-stage or smaller projects
    
    - **X and Y axes** are principal components that capture the most important variations in the data:
        - Projects closer together are more similar in their performance metrics
        - The X marker (✕) in each color shows the center of that cluster
    
    - **Hover over points** to see:
        - Project name
        - Grant amount
        - Cluster assignment
    """)

    # Key Insights section
    st.info(
        """
        **Key Insights**
                
        The data shows that **multiple clusters received a similar share of total funding, despite having different strengths and profiles**. For example:
        - The high-growth cluster (🟢) included breakout projects like DefiLlama and Hey.xyz, while the stable cluster (🔴) featured established infrastructure like L2BEAT and Dappnode.
        - The loyal donor cluster (🔵) included community-driven efforts such as EthStaker and Giveth.
        - The early-stage/smaller cluster (🟣) included innovative or emerging projects like POAPin and Hypercerts Foundation.
        
        **This inclusive distribution highlights that the round supported a broad spectrum of impact types—from foundational infrastructure to user-facing tools and community goods—affirming the round's commitment to diversity in OSS impact.**
        """
    )
    
    # Create scatter plot
    # First, let's verify the clusters
    print("Unique clusters:", viz_df['Cluster'].unique())
    print("Cluster counts:", viz_df['Cluster'].value_counts())
    
    # Create scatter plot with explicit color mapping
    fig = px.scatter(
        viz_df,
        x='PC1',
        y='PC2',
        color='Cluster',
        size='Grant Amount',
        hover_data=['Project', 'Grant Amount'],
        title='Project Performance Clusters',
        labels={
            'PC1': f'Principal Component 1 ({explained_variance[0]:.1%} variance)',
            'PC2': f'Principal Component 2 ({explained_variance[1]:.1%} variance)'
        },
        height=800,  # Increased height for better visualization
        text='Project'  # Add project names as text labels
    )
    
    # Force update the colors with a color map
    color_map = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple'
    }
    
    # Update colors for each trace
    for i, trace in enumerate(fig.data):
        if i < len(color_map):  # Only color the cluster traces
            trace.marker.color = [color_map[cluster] for cluster in viz_df['Cluster']]
    
    # Update text position and appearance
    fig.update_traces(
        textposition='top center',
        textfont=dict(size=10)
    )
    
    # Add cluster centers
    centers = pca.transform(kmeans.cluster_centers_)
    for i, center in enumerate(centers):
        fig.add_trace(
            go.Scatter(
                x=[center[0]],
                y=[center[1]],
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=15,
                    color=color_map[i]  # Match the cluster color
                ),
                name=f'Cluster {i} Center',
                showlegend=False
            )
        )
    
    # Display the plot
    st.plotly_chart(fig, use_container_width=True)

        # Add funding distribution analysis
    st.markdown("### Funding Distribution by Cluster")

    st.markdown("""
                This chart shows how funding is distributed across different project clusters. The colored bars represent the average grant amount received by each cluster, while the white line indicates each cluster's share of the total funding. This allows you to quickly compare which clusters received higher average grants and which contributed most to the overall funding pool, highlighting both the equity and concentration of funding among project types.
                """)
    
    # Calculate funding statistics per cluster
    funding_stats = pd.DataFrame({
        'Cluster': clusters,
        'Grant Amount': allocation_df['total']
    })
    
    cluster_funding = funding_stats.groupby('Cluster').agg({
        'Grant Amount': ['count', 'mean', 'sum', 'std']
    }).round(2)
    
    cluster_funding.columns = ['Number of Projects', 'Average Grant', 'Total Funding', 'Std Dev']
    
    
    # Calculate funding shares before visualization
    total_funding = cluster_funding['Total Funding'].sum()
    funding_shares = (cluster_funding['Total Funding'] / total_funding * 100).round(2)
    
    # Create a combined chart with two Y-axes
    fig_funding = go.Figure()
    
    # Add bar chart for average grant amount
    fig_funding.add_trace(
        go.Bar(
            x=cluster_funding.index,
            y=cluster_funding['Average Grant'],
            name='Average Grant Amount',
            marker_color=[color_map[cluster] for cluster in cluster_funding.index],
            yaxis='y1'
        )
    )
    
    # Add line chart for funding share percentage
    fig_funding.add_trace(
        go.Scatter(
            x=cluster_funding.index,
            y=funding_shares,
            name='% of Total Funding',
            mode='lines+markers',
            line=dict(color='white', width=2),
            marker=dict(size=10, symbol='diamond'),
            yaxis='y2'
        )
    )
    
    # Update layout with two Y-axes
    fig_funding.update_layout(
        title='Funding Distribution by Cluster',
        xaxis=dict(
            title='Cluster',
            tickmode='array',
            ticktext=[f'Cluster {i}' for i in cluster_funding.index],
            tickvals=cluster_funding.index,
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title='Average Grant Amount (USD)',
            titlefont=dict(color='white'),
            tickfont=dict(color='white')
        ),
        yaxis2=dict(
            title='% of Total Funding',
            titlefont=dict(color='white'),
            tickfont=dict(color='white'),
            overlaying='y',
            side='right',
            range=[0, max(funding_shares) * 1.1]  # Add 10% padding to top
        ),
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        ),
        height=600
    )
    
    st.plotly_chart(fig_funding, use_container_width=True)
    
    
    
    # Display cluster statistics
    st.markdown("### Cluster Statistics")
    
    # Calculate mean values for each metric by cluster
    cluster_stats = metrics_df[numeric_cols].copy()
    cluster_stats['Cluster'] = clusters
    cluster_stats['Grant Amount'] = allocation_df['total']
    
    # Calculate mean values for each cluster
    cluster_means = cluster_stats.groupby('Cluster').mean()
    
    # Define color mapping for consistency
    color_map = {
        0: 'red',
        1: 'blue',
        2: 'green',
        3: 'purple'
    }
    
    commentary = {
        0: "This cluster is characterized by moderate GMV growth and donor retention, with a relatively high number of active developers and a balanced grant amount. Projects in this cluster may represent stable, steadily growing initiatives with consistent community engagement.",
        1: "Projects in this cluster have lower GMV growth and fewer active developers, but show a high donor retention rate. This may indicate a smaller, but highly loyal donor base, possibly for niche or specialized projects.",
        2: "This cluster stands out for its very high GMV growth and unique donor growth, as well as the highest grant amounts. These projects are likely experiencing rapid expansion and attracting significant new support, possibly representing breakout or high-impact initiatives.",
        3: "Cluster 3 projects have the lowest GMV growth and grant amounts, with moderate donor and developer retention. These may be early-stage or smaller projects, or those in need of further support to scale."
    }
    
    # Display cluster statistics with color-coded headers and project lists
    for cluster in sorted(cluster_means.index):
        st.markdown(f"### Cluster {cluster} <span style='color:{color_map[cluster]}'>●</span>", unsafe_allow_html=True)
        st.markdown(f"{commentary[cluster]}")
        
        # List projects in this cluster
        project_names = metrics_df.loc[clusters == cluster, 'project'].tolist()
        st.markdown("**Projects in this cluster:** " + ", ".join(project_names) if project_names else "_No projects in this cluster_")
        
        st.dataframe(
            cluster_means.loc[[cluster]].style.format("{:.2f}"),
            use_container_width=True
        )
        
    st.markdown("""
    ---
    **Retrospective Notes (Draft)**

    - Continue supporting a wide range of project types (stable, high-growth, loyal-donor, and early-stage) to ensure a resilient and innovative OSS ecosystem.
    - Provide targeted resources for each cluster. Stable/Established Projects need reliable funding to maintain critical infrastructure, whereas Loyal Donor Base Projects need help to develop programs to scale and diversify their donor base.
    - Experiment with differentiated grant mechanisms to better meet the needs of diverse project types.
    """)

