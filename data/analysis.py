# Small Business Owners in the United States 🇺🇸
# This script analyzes the SCFP2019 dataset to explore small business ownership
# and performs clustering and PCA for high-variance financial features.

# ------------------------
# Import libraries
# ------------------------
import pandas as pd                 # Data manipulation
import plotly.express as px         # Interactive plots
import pickle                       # Save/load Python objects
from scipy.stats.mstats import trimmed_var  # Compute trimmed variance
from sklearn.cluster import KMeans  # K-Means clustering
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.metrics import silhouette_score  # Evaluate clustering
from sklearn.pipeline import make_pipeline, Pipeline  # Combine preprocessing + model
from sklearn.preprocessing import StandardScaler  # Standardize features
from sklearn.utils.validation import check_is_fitted  # Validate models
import warnings                      # Ignore warning messages

# Ignore warnings for cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# ------------------------
# Data wrangling
# ------------------------
def wrangle(filepath):
    """Load the dataset from a CSV file."""
    df = pd.read_csv(filepath)
    return df

# Load data
df = wrangle("data/SCFP2019.csv.gz")
print("df shape:", df.shape)
df.head()

# ------------------------
# Proportion of business owners
# ------------------------
prop_biz_owners = (df['HBUS'] == 1).sum() / len(df)
print("Proportion of business owners in df:", prop_biz_owners)

# Map INCCAT values to readable income ranges
inccat_dict = {
    1: "0-20",
    2: "21-39.9",
    3: "40-59.9",
    4: "60-79.9",
    5: "80-89.9",
    6: "90-100",
}

# ------------------------
# Count and proportion of INCCAT per HBUS
# ------------------------
# Count occurrences of each INCCAT per HBUS group
group_counts = df.groupby(['HBUS', 'INCCAT']).size().reset_index(name='count')

# Calculate proportion of each INCCAT within HBUS
group_totals = group_counts.groupby('HBUS')['count'].transform('sum')
group_counts['frequency'] = group_counts['count'] / group_totals

# Replace INCCAT numbers with readable labels
group_counts['INCCAT'] = group_counts['INCCAT'].map(inccat_dict)

# Keep relevant columns
df_inccat = group_counts[['HBUS', 'INCCAT', 'frequency']]

# Custom sorting of INCCAT frequencies for visualization
def custom_sort(sub_df):
    if sub_df['HBUS'].iloc[0] <= 5:
        return sub_df.sort_values(by='frequency', ascending=False)
    else:
        return sub_df.sort_values(by='frequency', ascending=True)

df_inccat = df_inccat.groupby('HBUS', group_keys=False).apply(custom_sort).reset_index(drop=True)

# ------------------------
# Visualization: Income distribution by HBUS
# ------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# Create figure
fig, ax = plt.subplots(figsize=(8, 4))

# Plot grouped bar chart
sns.barplot(
    data=df_inccat,
    x="INCCAT",
    y="frequency",
    hue="HBUS",
    order=list(inccat_dict.values()),
    ax=ax
)

# Add axis labels and title
ax.set_xlabel("Income Category", fontsize=12)
ax.set_ylabel("Frequency (%)", fontsize=12)
ax.set_title("Income Distribution: Business Owners vs. Non-Business Owners", fontsize=14)

# Customize legend
handles, _ = ax.get_legend_handles_labels()
ax.legend(
    handles=handles,
    title="HBUS",
    labels=["0", "1"],
    fontsize=10,
    title_fontsize=11
)

plt.tight_layout()
plt.show()

# ------------------------
# Filter small business owners with household income < 500k
# ------------------------
mask = (df["HBUS"] == 1) & (df["INCOME"] < 5e5)
df_small_biz = df[mask]
print("df_small_biz shape:", df_small_biz.shape)
df_small_biz.head()

# ------------------------
# Variance calculation
# ------------------------
# Calculate variance for all numeric columns
variances = df_small_biz.var(numeric_only=True)

# Top 10 features with largest variance
top_ten_var = variances.sort_values(ascending=True).tail(10)
top_ten_var

# Trimmed variance (ignores top/bottom 10% outliers)
top_ten_trim_var = df_small_biz.apply(trimmed_var, limits=(0.1, 0.1)).sort_values().tail(10)
top_ten_trim_var

# ------------------------
# Visualization: horizontal bar of trimmed variance
# ------------------------
fig = px.bar(
    x=top_ten_trim_var.values,
    y=top_ten_trim_var.index,
    orientation='h',
    title="Top 10 Features by Trimmed Variance"
)
fig.update_layout(xaxis_title="Trimmed Variance", yaxis_title="Feature")
fig.show()

# ------------------------
# Prepare data for clustering
# ------------------------
high_var_cols = top_ten_trim_var.tail(5).index.to_list()  # select top 5 high-variance features
X = df_small_biz[high_var_cols]
print("X shape:", X.shape)
X.head()

# ------------------------
# K-Means clustering: elbow & silhouette analysis
# ------------------------
n_clusters = range(2, 13)
inertia_errors = []
silhouette_scores = []

for k in n_clusters:
    pipeline = Pipeline([
        ('scaler', StandardScaler()),           # Standardize features
        ('kmeans', KMeans(n_clusters=k, random_state=42))  # K-Means
    ])
    
    pipeline.fit(X)
    kmeans_model = pipeline.named_steps['kmeans']
    
    inertia_errors.append(kmeans_model.inertia_)  # Sum of squared distances
    score = silhouette_score(X, kmeans_model.labels_)  # Clustering quality
    silhouette_scores.append(score)

print("Inertia:", inertia_errors[:11])
print("Silhouette Scores:", silhouette_scores[:3])

# Plot inertia vs number of clusters
fig = px.line(
    x=list(n_clusters),
    y=inertia_errors,
    markers=True,
    title="K-Means Model: Inertia vs Number of Clusters"
)
fig.update_layout(xaxis_title="Number of Clusters", yaxis_title="Inertia")
fig.show()

# Plot silhouette score vs number of clusters
fig = px.line(
    x=list(n_clusters),
    y=silhouette_scores,
    markers=True,
    title="K-Means Model: Silhouette Score vs Number of Clusters"
)
fig.update_layout(xaxis_title="Number of Clusters", yaxis_title="Silhouette Score")
fig.show()

# ------------------------
# Final K-Means model with 3 clusters
# ------------------------
final_model = make_pipeline(StandardScaler(), KMeans(n_clusters=3, random_state=42))
final_model.fit(X)

labels = final_model.named_steps["kmeans"].labels_
xgb = X.groupby(labels).mean()

# Side-by-side bar chart of clusters
fig = px.bar(
    xgb,
    barmode="group",
    title="Small Business Owner Finances by Cluster"
)
fig.update_layout(xaxis_title="Cluster", yaxis_title="Value [$]")
fig.show()

# ------------------------
# PCA for visualization
# ------------------------
pca = PCA(n_components=2, random_state=42)
X_t = pca.fit_transform(X)
X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])

print("X_pca shape:", X_pca.shape)
X_pca.head()

# Scatter plot of PCA components colored by cluster labels
fig = px.scatter(
    data_frame=X_pca,
    x="PC1",
    y="PC2",
    color=labels.astype(str),
    title="PCA Representation of Clusters"
)
fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")
fig.show()
