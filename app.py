import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config
st.set_page_config(page_title="Credit Card Customer Clustering", page_icon="💳", layout="wide")

# Custom CSS for premium UI
st.markdown("""
<style>
    .stApp {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1rem;
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    .stTextInput>div>div>input {
        background-color: #f9f9f9;
        border-radius: 10px;
    }
    .stSelectbox>div>div>select {
        background-color: #f9f9f9;
        border-radius: 10px;
    }
    .css-1aumxhk {
        background-color: #2c3e50;
        color: white;
        padding: 1rem;
        border-radius: 10px;
    }
    .css-1v0mbdj {
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        background-color: white;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stPlotlyChart {
        background-color: white;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        padding: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("Credit Card Customer Data.csv")
    df = df.drop("Sl_No", axis=1)
    df = df.drop_duplicates()
    df = df.dropna()
    return df

df = load_data()

# Sidebar navigation
st.sidebar.title("Navigation")
selected = st.sidebar.radio("Go to", ["Home", "EDA", "Clustering", "Recommendations"])

# Home Tab
if selected == "Home":
    st.header("Credit Card Customer Clustering App 💳")
    st.subheader("Welcome to our advanced clustering and recommendation system!")
    
    st.write("""
    This app allows you to explore credit card customer data, perform clustering analysis, 
    and receive personalized recommendations. Here's what you can do:
    
    1. **Explore the Dataset**: Learn about credit card customer attributes and trends.
    2. **Perform Clustering**: Compare different clustering algorithms and visualize results.
    3. **Get Recommendations**: Receive personalized credit card recommendations based on your preferences.
    
    Navigate through the app using the sidebar menu. Each section provides interactive visualizations 
    and insights to help you understand the data better.
    """)
    
    st.info("Start by exploring the Exploratory Data Analysis (EDA) tab to get familiar with the dataset!")

# EDA Tab
elif selected == "EDA":
    st.header("Exploratory Data Analysis 📊")
    
    # Data cleaning options
    st.subheader("Data Cleaning")
    col1, col2 = st.columns(2)
    with col1:
        if st.checkbox("Remove Duplicates"):
            df = df.drop_duplicates()
            st.success("Duplicates removed!")
    with col2:
        if st.checkbox("Handle Missing Values"):
            df = df.dropna()
            st.success("Missing values handled!")
    
    # Display basic statistics
    st.subheader("Dataset Overview")
    st.write(df.describe())
    
    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    corr = df.corr()
    fig = px.imshow(corr, color_continuous_scale='RdBu_r', zmin=-1, zmax=1)
    fig.update_layout(title="Feature Correlation Heatmap", height=600)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("The heatmap shows correlations between different features. Darker red indicates strong positive correlation, while darker blue indicates strong negative correlation.")
    
    # Distribution of credit limit
    st.subheader("Distribution of Average Credit Limit")
    fig = px.histogram(df, x="Avg_Credit_Limit", nbins=30, marginal="box", title="Distribution of Average Credit Limit")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("This histogram shows the distribution of average credit limits. The box plot on top provides additional information about the median and quartiles.")
    
    # Scatter plot of credit limit vs total credit cards
    st.subheader("Credit Limit vs Total Credit Cards")
    fig = px.scatter(df, x="Total_Credit_Cards", y="Avg_Credit_Limit", color="Total_visits_bank", 
                     hover_data=["Customer Key"], trendline="ols", title="Credit Limit vs Total Credit Cards")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("This scatter plot shows the relationship between the number of credit cards and the average credit limit. The color represents the total visits to the bank.")
    
    # Pie chart of customer distribution based on total visits
    st.subheader("Customer Distribution by Total Visits")
    visits_dist = df['Total_visits_bank'] + df['Total_visits_online']
    fig = px.pie(values=visits_dist.value_counts().values, names=visits_dist.value_counts().index, 
                 title="Distribution of Total Visits (Bank + Online)")
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    st.info("This pie chart shows the distribution of customers based on their total visits (both bank and online).")

# Clustering Tab
elif selected == "Clustering":
    st.header("Clustering Analysis and Comparison 🔬")
    
    # Feature selection
    st.subheader("Select Features for Clustering")
    features = st.multiselect("Choose features for clustering:", 
                              options=df.columns.tolist(),
                              default=["Avg_Credit_Limit", "Total_Credit_Cards", "Total_visits_bank", "Total_visits_online"])
    
    if len(features) < 2:
        st.warning("Please select at least two features for clustering.")
    else:
        X = df[features]
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # K-means clustering
        st.subheader("K-means Clustering")
        
        # Elbow method
        inertias = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(X_scaled)
            inertias.append(kmeans.inertia_)
        
        # Simple method to find elbow point
        diffs = np.diff(inertias)
        optimal_k = np.argmin(diffs) + 2  # Add 2 because we start from k=1 and diff reduces array size by 1
        
        fig_elbow = px.line(x=k_range, y=inertias, title="Elbow Method for K-means")
        fig_elbow.add_vline(x=optimal_k, line_dash="dash", line_color="red", annotation_text=f"Optimal k: {optimal_k}")
        fig_elbow.update_layout(height=400)
        st.plotly_chart(fig_elbow, use_container_width=True)
        
        n_clusters = st.slider("Select number of clusters for K-means:", min_value=2, max_value=10, value=optimal_k)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # DBSCAN clustering
        st.subheader("DBSCAN Clustering")
        col1, col2 = st.columns(2)
        with col1:
            eps = st.slider("Select eps value for DBSCAN:", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
        with col2:
            min_samples = st.slider("Select min_samples for DBSCAN:", min_value=2, max_value=20, value=5)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # Hierarchical clustering
        st.subheader("Hierarchical Clustering")
        n_clusters_hc = st.slider("Select number of clusters for Hierarchical Clustering:", min_value=2, max_value=10, value=optimal_k)
        hc = AgglomerativeClustering(n_clusters=n_clusters_hc)
        hc_labels = hc.fit_predict(X_scaled)
        
        # Visualizations
        def plot_3d_clusters(X, labels, title):
            fig = px.scatter_3d(X, x=features[0], y=features[1], z=features[2], color=labels,
                                title=title)
            fig.update_layout(height=600)
            return fig
        
        def calculate_metrics(X, labels):
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            davies = davies_bouldin_score(X, labels)
            return silhouette, calinski, davies
        
        tab1, tab2, tab3 = st.tabs(["K-means", "DBSCAN", "Hierarchical"])
        
        with tab1:
            fig_kmeans = plot_3d_clusters(X, kmeans_labels, "K-means Clustering Results")
            st.plotly_chart(fig_kmeans, use_container_width=True)
            
            sil_kmeans, cal_kmeans, dav_kmeans = calculate_metrics(X_scaled, kmeans_labels)
            col1, col2, col3 = st.columns(3)
            col1.metric("Silhouette Score", f"{sil_kmeans:.2f}")
            col2.metric("Calinski-Harabasz Score", f"{cal_kmeans:.2f}")
            col3.metric("Davies-Bouldin Score", f"{dav_kmeans:.2f}")
            
            st.info("The 3D scatter plot shows the clustering results of K-means. Each color represents a different cluster.")
        
        with tab2:
            fig_dbscan = plot_3d_clusters(X, dbscan_labels, "DBSCAN Clustering Results")
            st.plotly_chart(fig_dbscan, use_container_width=True)
            
            n_clusters_dbscan = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
            st.metric("Number of Clusters", n_clusters_dbscan)
            
            if len(set(dbscan_labels)) > 1:
                sil_dbscan, cal_dbscan, dav_dbscan = calculate_metrics(X_scaled, dbscan_labels)
                col1, col2, col3 = st.columns(3)
                col1.metric("Silhouette Score", f"{sil_dbscan:.2f}")
                col2.metric("Calinski-Harabasz Score", f"{cal_dbscan:.2f}")
                col3.metric("Davies-Bouldin Score", f"{dav_dbscan:.2f}")
            
            st.info("The 3D scatter plot shows the clustering results of DBSCAN. Each color represents a different cluster, with noise points in black (-1).")
        
        with tab3:
            fig_hc = plot_3d_clusters(X, hc_labels, "Hierarchical Clustering Results")
            st.plotly_chart(fig_hc, use_container_width=True)
            
            sil_hc, cal_hc, dav_hc = calculate_metrics(X_scaled, hc_labels)
            col1, col2, col3 = st.columns(3)
            col1.metric("Silhouette Score", f"{sil_hc:.2f}")
            col2.metric("Calinski-Harabasz Score", f"{cal_hc:.2f}")
            col3.metric("Davies-Bouldin Score", f"{dav_hc:.2f}")
            
            st.info("The 3D scatter plot shows the clustering results of Hierarchical Clustering. Each color represents a different cluster.")
        
        # Comparison
        st.subheader("Clustering Comparison")
        comparison_df = pd.DataFrame({
            'Algorithm': ['K-means', 'DBSCAN', 'Hierarchical'],
            'Number of Clusters': [n_clusters, n_clusters_dbscan, n_clusters_hc],
            'Silhouette Score': [sil_kmeans, sil_dbscan if len(set(dbscan_labels)) > 1 else 0, sil_hc],
            'Calinski-Harabasz Score': [cal_kmeans, cal_dbscan if len(set(dbscan_labels)) > 1 else 0, cal_hc],
            'Davies-Bouldin Score': [dav_kmeans, dav_dbscan if len(set(dbscan_labels)) > 1 else 0, dav_hc]
        })
        
        fig = go.Figure(data=[go.Table(
            header=dict(values=list(comparison_df.columns),
                        fill_color='#2c3e50',
                        align='left',
                        font=dict(color='white', size=12)),
            cells=dict(values=[comparison_df[k].tolist() for k in comparison_df.columns],
                       fill_color='#f0f2f6',
                       align='left'))
        ])
        fig.update_layout(title="Clustering Algorithm Comparison", height=300)
        st.plotly_chart(fig, use_container_width=True)
        
        # Insights
        st.subheader("Clustering Insights")
        best_algorithm = comparison_df.loc[comparison_df['Silhouette Score'].idxmax(), 'Algorithm']
        st.success(f"Based on the silhouette score, the best performing algorithm is: {best_algorithm}")
        
        if best_algorithm == 'K-means':
            st.write("K-means performed well, suggesting that the clusters are relatively spherical and of similar sizes.")
        elif best_algorithm == 'DBSCAN':
            st.write("DBSCAN performed well, indicating that the clusters have varying densities and shapes.")
        else:
            st.write("Hierarchical clustering performed well, suggesting that there might be a meaningful hierarchy in the data structure.")

# Recommendation Tab
elif selected == "Recommendations":
    st.header("Credit Card Recommendations 💳")
    
    st.write("Based on your preferences, we'll recommend credit cards that might suit you best.")
    
    # User preferences
    st.subheader("Your Preferences")
    col1, col2, col3 = st.columns(3)
    with col1:
        preferred_credit_limit = st.slider("Preferred Credit Limit", min_value=0, max_value=200000, value=50000, step=1000)
    with col2:
        preferred_cards = st.slider("Number of Credit Cards", min_value=1, max_value=10, value=3)
    with col3:
        preferred_visits = st.radio("Preferred Mode of Interaction", options=["Mostly Online", "Mostly Bank Visits", "Balanced"])
    
    # Simple recommendation system
    def recommend_cards(credit_limit, num_cards, visit_preference):
        if visit_preference == "Mostly Online":
            df_filtered = df[df['Total_visits_online'] > df['Total_visits_bank']]
        elif visit_preference == "Mostly Bank Visits":
            df_filtered = df[df['Total_visits_bank'] > df['Total_visits_online']]
        else:
            df_filtered = df
        
        df_filtered['score'] = (
            (1 - abs(df_filtered['Avg_Credit_Limit'] - credit_limit) / credit_limit) * 0.5 +
            (1 - abs(df_filtered['Total_Credit_Cards'] - num_cards) / num_cards) * 0.5
        )
        
        return df_filtered.nlargest(5, 'score')
    
    recommendations = recommend_cards(preferred_credit_limit, preferred_cards, preferred_visits)
    
    st.subheader("Top 5 Recommended Credit Cards")
    for i, row in recommendations.iterrows():
        with st.expander(f"Card {i+1} - Credit Limit: ${row['Avg_Credit_Limit']:,.0f}"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Credit Cards", row['Total_Credit_Cards'])
                st.metric("Bank Visits", row['Total_visits_bank'])
            with col2:
                st.metric("Online Visits", row['Total_visits_online'])
                st.metric("Total Calls Made", row['Total_calls_made'])
            with col3:
                similarity_score = row['score'] * 100
                st.metric("Similarity Score", f"{similarity_score:.2f}%")
    
    # Data-driven insights
    st.subheader("Recommendation Insights")
    avg_credit_limit = recommendations['Avg_Credit_Limit'].mean()
    avg_total_cards = recommendations['Total_Credit_Cards'].mean()
    avg_online_visits = recommendations['Total_visits_online'].mean()
    avg_bank_visits = recommendations['Total_visits_bank'].mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Average Credit Limit", f"${avg_credit_limit:,.2f}")
        st.metric("Average Number of Credit Cards", f"{avg_total_cards:.2f}")
    with col2:
        st.metric("Average Online Visits", f"{avg_online_visits:.2f}")
        st.metric("Average Bank Visits", f"{avg_bank_visits:.2f}")
    
    st.write("### Key Takeaways")
    takeaways = []
    if avg_credit_limit > preferred_credit_limit:
        takeaways.append(f"The recommended cards have a higher average credit limit (${avg_credit_limit:,.2f}) than your preference (${preferred_credit_limit:,.2f}). This suggests you might qualify for higher credit limits.")
    else:
        takeaways.append(f"The recommended cards have a lower average credit limit (${avg_credit_limit:,.2f}) than your preference (${preferred_credit_limit:,.2f}). You might want to consider building your credit score to qualify for higher limits.")
    
    if avg_total_cards > preferred_cards:
        takeaways.append(f"On average, customers similar to you tend to have more credit cards ({avg_total_cards:.2f}) than your preference ({preferred_cards}). Consider if having additional cards aligns with your financial goals.")
    else:
        takeaways.append(f"The recommended number of credit cards ({avg_total_cards:.2f}) aligns closely with your preference ({preferred_cards}).")
    
    if preferred_visits == "Mostly Online":
        if avg_online_visits > avg_bank_visits:
            takeaways.append("The recommendations align well with your preference for online interactions.")
        else:
            takeaways.append("Despite your preference for online interactions, the recommendations show a tendency for more bank visits. You might want to explore online-friendly card options.")
    elif preferred_visits == "Mostly Bank Visits":
        if avg_bank_visits > avg_online_visits:
            takeaways.append("The recommendations align well with your preference for bank visits.")
        else:
            takeaways.append("Despite your preference for bank visits, the recommendations show a tendency for more online interactions. You might benefit from cards with strong branch support.")
    else:
        if abs(avg_online_visits - avg_bank_visits) < 1:
            takeaways.append("The recommendations show a good balance between online and bank interactions, aligning with your preference.")
        else:
            takeaways.append(f"The recommendations show a slight preference for {'online' if avg_online_visits > avg_bank_visits else 'bank'} interactions. Consider if this aligns with your needs.")
    
    for takeaway in takeaways:
        st.info(takeaway)
    
    st.write("These recommendations are based on your preferred credit limit, number of cards, and interaction mode. The system finds customers with similar profiles to provide these suggestions.")
    
    # Feedback
    st.subheader("Feedback")
    col1, col2 = st.columns(2)
    with col1:
        feedback = st.slider("How would you rate these recommendations?", min_value=1, max_value=5, value=3)
    with col2:
        comments = st.text_area("Any additional comments?")
    
    if st.button("Submit Feedback"):
        st.success("Thank you for your feedback! We'll use it to improve our recommendations.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; padding: 10px;">
        <p>Created by RS | Data Source: Credit Card Customer Dataset</p>
        <p>© 2023 Credit Card Clustering App. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)
