Small Business Owners Analysis

Unsupervised Learning Project | K-Means Clustering & PCA Visualization

This project analyzes small business ownership in the United States using the SCFP2019 dataset. By applying unsupervised learning techniques, we explore financial patterns among business owners, identify distinct clusters, and visualize high-variance financial features.

🔹 Project Overview

The goal of this project is to explore and understand patterns among small business owners in the US. Specifically, it:

Examines the proportion of business owners across income categories.

Identifies the top high-variance financial features.

Performs K-Means clustering to group similar business owners.

Uses PCA (Principal Component Analysis) to reduce dimensionality for visualization.

Produces interactive and static visualizations to gain actionable insights.

Note: This is an unsupervised learning project, meaning there is no target variable. The focus is on discovering patterns and clusters in the data rather than predicting outcomes.

🔹 Key Techniques & Libraries

Python & Pandas – Data manipulation and preprocessing

Plotly & Seaborn – Interactive and static visualizations

Scipy – Trimmed variance for robust feature selection

Scikit-learn –

StandardScaler for feature normalization

K-Means clustering for grouping similar business owners

PCA for dimensionality reduction and visualization

Silhouette score for evaluating clustering quality

Pipeline – Combines scaling and clustering steps efficiently

🔹 Highlights

Income Distribution: Compares income ranges between business owners and non-owners.

Feature Selection: Identifies features with highest variance for clustering.

Clustering:

Uses elbow method and silhouette scores to find optimal clusters.

Segments business owners into 3 meaningful clusters.

PCA Visualization: Displays clusters in 2D for easy interpretation.
