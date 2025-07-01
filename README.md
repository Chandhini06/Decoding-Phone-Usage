# Decoding-Phone-Usage

**Approach**

1) **Data Preparation:**

    Used a dataset containing user IDs, device models, operating systems, and usage statistics. Merged and preprocessed the dataset to ensure consistency and accuracy.

2) **Data Cleaning:**

    Handled missing values using imputation techniques. Standardized formats for features like operating systems and device models. Removed outliers based on statistical thresholds.

3) **Exploratory Data Analysis (EDA):**

    Analyzed trends in mobile app usage, screen-on time, and battery consumption. Visualized correlations between features like data usage and battery drain. Identified patterns in Primary use Class.

4) **Machine Learning and Clustering:**

    Implemented classification models to predict Primary use Class:
    
    - Logistic Regression Decision Trees Random Forest Gradient Boosting (e.g., XGBoost or LightGBM) etc. Applied clustering techniques to group users based on device usage patterns:

    - K-Means Hierarchical Clustering DBSCAN Gaussian Mixture Models Spectral Clustering etc. Evaluate classification models using metrics like precision, recall, and accuracy.

    - Analyzed clustering performance using silhouette scores and visualization techniques.

6) **Application Development:**

    Build a user-friendly interface using Streamlit to: Displayed visualizations and insights from EDA. Allowed users to input data for primary use classification. Present clustering results and user segmentation.

7) **Deployment:**

    Deployed the Streamlit application for accessibility and user interaction.
