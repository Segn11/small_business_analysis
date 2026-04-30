# Small Business Owners Analysis (SCFP 2019)

**Unsupervised Learning Project — K-Means Clustering & PCA Visualization**

This project explores **financial patterns among small business owners in the United States** using the **SCFP 2019** dataset. By applying unsupervised learning techniques, we identify distinct groups of business owners, highlight the most informative (high-variance) financial features, and visualize the resulting clusters with PCA.

---

## Project Goals

This analysis is designed to **discover structure in the data** (no target variable), focusing on:

- **Profiling ownership patterns** across income categories  
- **Selecting key financial features** using robust variance-based methods  
- **Clustering** similar business owners with **K-Means**  
- **Reducing dimensionality** with **PCA** for interpretable 2D visualization  
- Generating **interactive and static visuals** to communicate insights clearly

> Note: This is an **unsupervised learning** project—its purpose is to uncover patterns and segments, not to predict outcomes.

---

## What’s Inside

### 1) Income Distribution Analysis
Compare income ranges between:
- **Business owners**
- **Non-owners**

This helps frame how ownership relates to financial standing.

### 2) High-Variance Feature Discovery
To focus clustering on informative signals, the project identifies **top financial features by variance**, using **trimmed variance** for robustness against outliers.

### 3) Clustering (K-Means)
Business owners are segmented into meaningful groups:
- Uses the **Elbow Method** to estimate a reasonable `k`
- Uses **Silhouette Score** to evaluate clustering quality
- Final segmentation identifies **3 interpretable clusters**

### 4) PCA Visualization
PCA is used to project clustered points into **2D space**, making cluster separation and structure easier to interpret and present.

---

## Tools & Libraries

- **Python** — analysis and modeling  
- **Pandas** — data cleaning and preprocessing  
- **Plotly** — interactive visualizations  
- **Seaborn / Matplotlib** — static plots  
- **SciPy** — trimmed variance for robust feature selection  
- **Scikit-learn**  
  - `StandardScaler` — feature normalization  
  - `KMeans` — clustering  
  - `PCA` — dimensionality reduction  
  - `silhouette_score` — cluster quality evaluation  
  - `Pipeline` — clean modeling workflow (scaling → clustering)

---

## Typical Workflow

1. Load and clean SCFP 2019 data  
2. Explore ownership & income distributions  
3. Select high-variance financial features  
4. Scale features (`StandardScaler`)  
5. Cluster using `KMeans` (choose `k` using elbow + silhouette)  
6. Visualize clusters using PCA (2D)  
7. Interpret clusters and summarize insights

---

## Results (High-Level)

- Clear differences emerge when comparing **income distribution** between owners and non-owners  
- Variance-based filtering isolates the most informative financial dimensions  
- K-Means forms **3 distinct segments** of business owners  
- PCA visualization provides an intuitive view of **cluster separation and overlap**

---

## Future Improvements (Optional Roadmap)

- Try alternative clustering methods (GMM, DBSCAN, Hierarchical)
- Add cluster profiling tables (median values per cluster for interpretability)
- Automate reporting (export plots + summary into a single report notebook)
- Add reproducible environment setup (`requirements.txt` / `environment.yml`)

---

## Contributing

Suggestions and improvements are welcome:
- Open an issue for bugs or enhancements
- Submit a pull request with clear descriptions and results

---

## License

Add a license if you plan to share or reuse this project publicly (e.g., MIT, Apache-2.0).

---
**Keywords:** Unsupervised Learning, K-Means, PCA, Segmentation, SCFP 2019, Small Business, Financial Analysis
