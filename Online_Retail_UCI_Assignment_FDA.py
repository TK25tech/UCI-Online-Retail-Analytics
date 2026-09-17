# =======================================================
# CUSTOMER SEGMENTATION + VALUE PREDICTION PROJECT
# Online Retail Dataset
# =======================================================

# Business Context:
# UK-based online retailer selling giftware and decorative products.
# Objective: understand customer personas, segment customers, and
# identify/predict high-value customers so marketing spend can be targeted.

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from scipy.spatial.distance import euclidean

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, silhouette_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

# ---- Visualisation style ----
sns.set_theme(style="darkgrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["axes.titlesize"] = 16
plt.rcParams["axes.labelsize"] = 12
pd.set_option("display.max_columns", None)

# ---- Reproducibility / output paths ----
RANDOM_STATE = 42
# Put Online Retail.xlsx beside this script, or set ONLINE_RETAIL_PATH.
DATA_PATH = os.getenv("ONLINE_RETAIL_PATH", r"C:\Users\kampa\Documents\FDA- Dr. Shafi\Online Retail.xlsx")
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def save_fig(name):
    """Save the current matplotlib figure as a report-ready PNG."""
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}.png"), dpi=300, bbox_inches="tight")


def save_plotly(fig, name):
    """Save a Plotly figure as a static PNG (needs: pip install -U kaleido)."""
    try:
        fig.write_image(os.path.join(OUTPUT_DIR, f"{name}.png"))
    except Exception as e:
        print(f"Could not save '{name}' as an image â€” install kaleido (pip install -U kaleido): {e}")


# =======================================================
# Phase 1 - Business Understanding
# =======================================================
# RQ1: Can customers be grouped into meaningful segments? (unsupervised)
# RQ2: What distinguishes high-value segments from others? (profiling)
# RQ3: Can earlier purchasing behaviour predict future customer value? (supervised)

# =======================================================
# Loading Dataset
# =======================================================
df = pd.read_excel(DATA_PATH)

print(f"Dataset Shape: {df.shape}")
print(f"\nData Types:\n{df.dtypes}")
print(f"\nFirst 5 Rows:\n{df.head()}")
df.info()
print(f"\nStatistical Summary:\n{df.describe(include='all')}")

# Basic data dictionary and variable classification required by Task 1.
data_dictionary = pd.DataFrame({
    "Variable": ["InvoiceNo", "StockCode", "Description", "Quantity", "InvoiceDate",
                 "UnitPrice", "CustomerID", "Country"],
    "Definition": ["Transaction/order identifier", "Product identifier", "Product name",
                   "Units on the transaction line", "Transaction date and time",
                   "Price per unit in GBP", "Customer identifier", "Customer country"],
    "AnalyticalType": ["Categorical", "Categorical", "Categorical", "Numerical",
                       "Temporal", "Numerical", "Categorical", "Categorical"],
})
print(f"\nBasic Data Dictionary:\n{data_dictionary.to_string(index=False)}")
print("\nOrdinal variables: none of the original fields has a natural ranked order.")

# =======================================================
# Phase 2 - Data Understanding
# =======================================================
print(f"\nMissing Values:\n{df.isnull().sum()}")
print(f"\nDuplicate Rows: {df.duplicated().sum()}")

cancelled_orders = df["InvoiceNo"].astype(str).str.startswith("C").sum()
print(f"\nCancelled Orders: {cancelled_orders}")
print(f"Negative Quantity Rows: {(df['Quantity'] <= 0).sum()}")
print(f"Invalid Unit Price Rows: {(df['UnitPrice'] <= 0).sum()}")

# =======================================================
# Phase 3 - Data Cleaning
# =======================================================
original_rows = len(df)
cleaning_stages = [("Original", original_rows)]
missing_customerID_count = df["CustomerID"].isnull().sum()
missing_customerID_percentage = missing_customerID_count / original_rows * 100
print(f"\nMissing CustomerIDs: {missing_customerID_count} "
      f"({missing_customerID_percentage:.2f}% of original data)")

before = len(df)
df = df.dropna(subset=["CustomerID"])
print(f"Missing CustomerID Removed: {before - len(df)}")
cleaning_stages.append(("After customer-ID filter", len(df)))

before = len(df)
df = df.drop_duplicates()
print(f"Duplicate Rows Removed: {before - len(df)}")
cleaning_stages.append(("After duplicate filter", len(df)))

before = len(df)
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
print(f"Cancelled Orders Removed: {before - len(df)}")
cleaning_stages.append(("After cancellation filter", len(df)))

before = len(df)
df = df[df["Quantity"] > 0]
print(f"Negative Quantity Removed: {before - len(df)}")
cleaning_stages.append(("After quantity filter", len(df)))

before = len(df)
df = df[df["UnitPrice"] > 0]
print(f"Invalid Price Removed: {before - len(df)}")
cleaning_stages.append(("Final cleaned", len(df)))

df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

final_rows = len(df)
print(f"\nFinal Rows: {final_rows}")
print(f"Rows Removed: {original_rows - final_rows}")
print(f"Data Retained: {(final_rows / original_rows) * 100:.2f}%")
cleaning_summary = pd.DataFrame(cleaning_stages, columns=["Stage", "Rows"])
print(f"\nCleaning Summary:\n{cleaning_summary.to_string(index=False)}")
cleaning_summary.to_csv(os.path.join(OUTPUT_DIR, "cleaning_summary.csv"), index=False)

# =======================================================
# Phase 4 - Feature Engineering
# =======================================================
# Two dataframes are kept alive for the rest of the project:
#   df_transactions -> invoice-level, feeds regression + revenue/product/country charts
#   df_customers    -> one row per customer (RFM), feeds clustering + classification

df_transactions = df.copy()
df_transactions["TotalPrice"] = df_transactions["Quantity"] * df_transactions["UnitPrice"]
df_transactions["InvoiceMonth"] = df_transactions["InvoiceDate"].dt.month
df_transactions["InvoiceDay"] = df_transactions["InvoiceDate"].dt.day_name()
df_transactions["InvoiceHour"] = df_transactions["InvoiceDate"].dt.hour

reference_date = df_transactions["InvoiceDate"].max() + pd.Timedelta(days=1)

df_customers = (
    df_transactions
    .groupby("CustomerID")
    .agg(
        Recency=("InvoiceDate", lambda x: (reference_date - x.max()).days),
        Frequency=("InvoiceNo", "nunique"),
        Monetary=("TotalPrice", "sum"),
        TotalQuantity=("Quantity", "sum"),
    )
    .reset_index()
)
# NOTE: a 'NumberOfOrders' column existed in the original script using the exact
# same aggregation as 'Frequency' ('InvoiceNo','nunique') â€” removed as duplicate/unused.

# --- AverageBasketSize, corrected ---
# Original version averaged line-item revenue (TotalPrice mean), which is NOT a
# basket/order value and is also a near-circular summary of the same values that
# sum to Monetary. Correct definition: total revenue per ORDER, then averaged
# per customer across their orders.
order_value = (
    df_transactions
    .groupby(["CustomerID", "InvoiceNo"])["TotalPrice"]
    .sum()
    .reset_index()
)
avg_basket = (
    order_value.groupby("CustomerID")["TotalPrice"]
    .mean()
    .rename("AverageBasketSize")
    .reset_index()
)
df_customers = df_customers.merge(avg_basket, on="CustomerID")

# --- Demonstrated categorical encoding (Task 2 requirement) ---
# Each customer's most common shipping country, encoded as a simple binary flag.
# Not forced into the existing regression/classification feature lists below â€”
# available if you choose to strengthen those models with it.
customer_country = (
    df_transactions.groupby("CustomerID")["Country"]
    .agg(lambda x: x.mode().iloc[0])
    .rename("Country")
)
df_customers = df_customers.merge(customer_country, on="CustomerID")
df_customers["IsUK"] = (df_customers["Country"] == "United Kingdom").astype(int)

feature_dictionary = pd.DataFrame({
    "Feature": ["TotalPrice", "InvoiceMonth", "InvoiceDay", "InvoiceHour", "Recency",
                "Frequency", "Monetary", "TotalQuantity", "AverageBasketSize", "IsUK"],
    "Definition": ["Quantity multiplied by UnitPrice", "Transaction month number",
                   "Transaction weekday", "Transaction hour", "Days since latest purchase",
                   "Number of unique invoices", "Total customer revenue",
                   "Total units purchased", "Mean order revenue", "1 for UK; 0 otherwise"],
})
print(f"\nEngineered Feature Dictionary:\n{feature_dictionary.to_string(index=False)}")
feature_dictionary.to_csv(os.path.join(OUTPUT_DIR, "engineered_features.csv"), index=False)

print(f"\nTransaction Dataset:\n{df_transactions.head()}")
print(f"\nCustomer Dataset:\n{df_customers.head()}")
print(f"\nTransaction Shape: {df_transactions.shape}")
print(f"Customer Shape: {df_customers.shape}")

# Full descriptive-statistics evidence required by Task 1.
def descriptive_statistics(frame, columns):
    rows = []
    for column in columns:
        series = frame[column].dropna()
        q1, q3 = series.quantile([0.25, 0.75])
        modes = series.mode()
        rows.append({
            "Variable": column,
            "Count": series.count(),
            "Mean": series.mean(),
            "Median": series.median(),
            "Mode": modes.iloc[0] if not modes.empty else np.nan,
            "Minimum": series.min(),
            "Maximum": series.max(),
            "Range": series.max() - series.min(),
            "Variance": series.var(ddof=1),
            "StdDev": series.std(ddof=1),
            "Q1": q1,
            "Q3": q3,
            "IQR": q3 - q1,
        })
    return pd.DataFrame(rows)


descriptive_table = descriptive_statistics(
    df_customers,
    ["Recency", "Frequency", "Monetary", "AverageBasketSize", "TotalQuantity"],
)
print(f"\nComplete Customer Descriptive Statistics:\n{descriptive_table.to_string(index=False)}")
descriptive_table.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"), index=False)

# =======================================================
# Phase 5 - Preprocessing & Mathematical Foundations
# =======================================================
print(f"\nSkewness Before Transformation:\n{df_customers[['Monetary', 'Frequency']].skew()}")

df_customers["Monetary_Log"] = np.log1p(df_customers["Monetary"])
df_customers["Frequency_Log"] = np.log1p(df_customers["Frequency"])

print(f"\nSkewness After Transformation:\n{df_customers[['Monetary_Log', 'Frequency_Log']].skew()}")

# Standardisation formula used here: z = (x - mean(x)) / std(x)
# Required because K-means (Phase 8) relies on Euclidean distance, which is
# scale-sensitive â€” without this, Monetary would dominate purely because its
# numeric range is far larger than Recency's.
scaler = StandardScaler()
features_to_scale = ["Recency", "Frequency_Log", "Monetary_Log", "TotalQuantity"]
scaled_features = scaler.fit_transform(df_customers[features_to_scale])

df_scaled = pd.DataFrame(scaled_features, columns=features_to_scale)
df_scaled.insert(0, "CustomerID", df_customers["CustomerID"].values)
print(f"\nScaled Dataset:\n{df_scaled.head()}")

# Euclidean distance demo between two customers (post-scaling)
customer1 = df_scaled.iloc[0, 1:]
customer2 = df_scaled.iloc[1, 1:]
distance = euclidean(customer1, customer2)
print(f"\nEuclidean Distance Between Two Customers: {distance:.3f}")

fig, ax = plt.subplots(1, 2, figsize=(14, 5))
sns.histplot(df_customers["Monetary"], bins=50, ax=ax[0])
ax[0].set_title("Customer Spending Before Log Transformation")
ax[0].set_xlabel("Monetary Value")
sns.histplot(df_customers["Monetary_Log"], bins=50, ax=ax[1])
ax[1].set_title("Customer Spending After Log Transformation")
ax[1].set_xlabel("Log Monetary Value")
plt.tight_layout()
save_fig("monetary_log_transform")
plt.show()

# =======================================================
# Phase 6 - Exploratory Data Analysis
# =======================================================
daily_sales = df_transactions.groupby(df_transactions["InvoiceDate"].dt.date)["TotalPrice"].sum()
fig = px.line(daily_sales, title="Daily Revenue Trend",
              labels={"value": "Revenue (GBP)", "date": "Date"}, markers=True)
fig.update_layout(template="plotly_dark", title_x=0.5, showlegend=False)
save_plotly(fig, "daily_revenue_trend")
fig.show()

top_products = (
    df_transactions.groupby("Description")["TotalPrice"].sum()
    .sort_values(ascending=False).head(10).sort_values()
)
fig = px.bar(top_products, orientation="h", title="Top 10 Products by Revenue",
             labels={"value": "Revenue (GBP)", "Description": "Product"})
fig.update_layout(template="plotly_dark", title_x=0.5, showlegend=False)
save_plotly(fig, "top_products")
fig.show()

country_sales = (
    df_transactions.groupby("Country")["TotalPrice"].sum()
    .sort_values(ascending=False).head(10).sort_values()
)
fig = px.bar(country_sales, orientation="h", title="Top 10 Countries by Revenue",
             labels={"value": "Revenue (GBP)", "Country": "Country"})
fig.update_layout(template="plotly_dark", title_x=0.5, showlegend=False)
save_plotly(fig, "top_countries")
fig.show()

fig = px.histogram(df_customers, x="Monetary", nbins=50, title="Customer Spending Distribution")
fig.update_layout(template="plotly_dark", title_x=0.5)
save_plotly(fig, "spending_distribution")
fig.show()

fig = px.histogram(df_customers, x="Frequency", nbins=30, title="Customer Purchase Frequency")
fig.update_layout(template="plotly_dark", title_x=0.5)
save_plotly(fig, "frequency_distribution")
fig.show()

corr = df_customers.drop(columns=["CustomerID", "Country"]).corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, fmt=".2f", linewidths=0.5)
plt.title("Customer Behaviour Correlation Matrix")
save_fig("correlation_heatmap")
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for axis, column in zip(axes, ["Recency", "Frequency", "Monetary"]):
    sns.boxplot(y=df_customers[column], ax=axis)
    axis.set_title(column)
    axis.set_ylabel(column)
fig.suptitle("Customer Feature Outlier Analysis (Separate Scales)")
plt.tight_layout()
save_fig("outlier_boxplot")
plt.show()

# =======================================================
# Phase 7 - Leakage-Safe Future Customer Value Prediction
# =======================================================
# Predictors are constructed only from transactions BEFORE the cutoff; the
# continuous and categorical targets use transactions ON/AFTER the cutoff.
# This avoids using same-period summaries to predict their own arithmetic total.
PREDICTION_CUTOFF = pd.Timestamp("2011-09-01")
observation = df_transactions[df_transactions["InvoiceDate"] < PREDICTION_CUTOFF].copy()
future = df_transactions[df_transactions["InvoiceDate"] >= PREDICTION_CUTOFF].copy()
print(f"\nObservation Period: {observation['InvoiceDate'].min()} to {observation['InvoiceDate'].max()}")
print(f"Prediction Period: {future['InvoiceDate'].min()} to {future['InvoiceDate'].max()}")

historical_features = (
    observation.groupby("CustomerID")
    .agg(
        HistRecency=("InvoiceDate", lambda x: (PREDICTION_CUTOFF - x.max()).days),
        HistFrequency=("InvoiceNo", "nunique"),
        HistMonetary=("TotalPrice", "sum"),
        HistTotalQuantity=("Quantity", "sum"),
    )
    .reset_index()
)
historical_order_values = (
    observation.groupby(["CustomerID", "InvoiceNo"])["TotalPrice"].sum().reset_index()
)
historical_baskets = (
    historical_order_values.groupby("CustomerID")["TotalPrice"]
    .mean().rename("HistAverageBasketSize").reset_index()
)
historical_country = (
    observation.groupby("CustomerID")["Country"]
    .agg(lambda x: x.mode().iloc[0]).rename("Country").reset_index()
)
historical_features = historical_features.merge(historical_baskets, on="CustomerID")
historical_features = historical_features.merge(historical_country, on="CustomerID")
historical_features["IsUK"] = (historical_features["Country"] == "United Kingdom").astype(int)

future_value = future.groupby("CustomerID")["TotalPrice"].sum().rename("FutureMonetary")
model_data = historical_features.merge(future_value, on="CustomerID", how="left")
model_data["FutureMonetary"] = model_data["FutureMonetary"].fillna(0)

predictor_columns = [
    "HistRecency", "HistFrequency", "HistMonetary", "HistTotalQuantity",
    "HistAverageBasketSize", "IsUK",
]
X = model_data[predictor_columns]
y = model_data["FutureMonetary"]
print(f"\nTemporal Modelling Dataset Shape: {model_data.shape}")
print(f"Customers With No Future Purchase: {(y == 0).sum()} ({(y == 0).mean():.2%})")
print(f"Feature Matrix X Shape: {X.shape}; Target Vector y Shape: {y.shape}")
print("Linear algebra notation: X is an n x p feature matrix and y is an n x 1 target vector.")


def regression_metrics(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "MSE": mse,
        "RMSE": np.sqrt(mse),
        "R2": r2_score(actual, predicted),
    }


def evaluate_regressors(features, target, test_size, split_label):
    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=test_size, random_state=RANDOM_STATE
    )
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, min_samples_leaf=3, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    rows, fitted = [], {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        train_prediction = model.predict(X_train)
        test_prediction = model.predict(X_test)
        for sample, actual, predicted in [
            ("Train", y_train, train_prediction), ("Test", y_test, test_prediction)
        ]:
            rows.append({"Split": split_label, "Sample": sample, "Model": name,
                         **regression_metrics(actual, predicted)})
        fitted[name] = {
            "model": model, "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "train_prediction": train_prediction, "test_prediction": test_prediction,
        }
    return pd.DataFrame(rows), fitted


results_80, fitted_80 = evaluate_regressors(X, y, 0.20, "80/20")
results_70, fitted_70 = evaluate_regressors(X, y, 0.30, "70/30")
regression_results = pd.concat([results_80, results_70], ignore_index=True)
print(f"\nLeakage-Safe Regression Performance:\n{regression_results.to_string(index=False)}")
regression_results.to_csv(os.path.join(OUTPUT_DIR, "regression_results.csv"), index=False)

rf_fit = fitted_80["Random Forest"]
lr_fit = fitted_80["Linear Regression"]
rf_model = rf_fit["model"]
y_test80 = rf_fit["y_test"]
rf_prediction = rf_fit["test_prediction"]

plt.figure(figsize=(8, 7))
sns.scatterplot(x=y_test80, y=rf_prediction, alpha=0.65)
line_min = min(y_test80.min(), rf_prediction.min())
line_max = max(y_test80.max(), rf_prediction.max())
plt.plot([line_min, line_max], [line_min, line_max], linestyle="--", color="red")
plt.xlabel("Actual Future Customer Value (GBP)")
plt.ylabel("Predicted Future Customer Value (GBP)")
plt.title("Random Forest: Actual vs Predicted Future Customer Value")
save_fig("actual_vs_predicted_future_value")
plt.show()

# Linear Regression diagnostics directly address its assumptions.
lr_residuals = lr_fit["y_test"] - lr_fit["test_prediction"]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
sns.scatterplot(x=lr_fit["test_prediction"], y=lr_residuals, alpha=0.6, ax=axes[0])
axes[0].axhline(0, linestyle="--", color="red")
axes[0].set_xlabel("Linear Regression Predicted Value (GBP)")
axes[0].set_ylabel("Residual (Actual - Predicted)")
axes[0].set_title("Residuals vs Fitted Values")
sns.histplot(lr_residuals, bins=40, kde=True, ax=axes[1])
axes[1].set_title("Linear Regression Residual Distribution")
axes[1].set_xlabel("Residual (GBP)")
plt.tight_layout()
save_fig("linear_regression_residual_diagnostics")
plt.show()

rf_importance = pd.Series(rf_model.feature_importances_, index=predictor_columns).sort_values(ascending=False)
print(f"\nRandom Forest Feature Importance:\n{rf_importance}")
lr_coefficients = pd.Series(lr_fit["model"].coef_, index=predictor_columns).sort_values(ascending=False)
print(f"\nLinear Regression Coefficients:\n{lr_coefficients}")
# MSE loss: J(beta) = (1/n) * sum((y_i - x_i beta)^2).
# Its gradient is dJ/dbeta = (-2/n) * X.T * (y - X beta); optimisation moves
# coefficients opposite the gradient until the loss can no longer be reduced.

# =======================================================
# Phase 8 - Customer Segmentation (Clustering)
# =======================================================
features = df_scaled[["Recency", "Frequency_Log", "Monetary_Log", "TotalQuantity"]]

k_range = range(2, 11)
inertia, silhouette_scores = [], []
for k in k_range:
    model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
    labels = model.fit_predict(features)  # single fit reused for both metrics
    inertia.append(model.inertia_)
    silhouette_scores.append(silhouette_score(features, labels))

cluster_selection = pd.DataFrame({"Clusters": list(k_range), "Inertia": inertia,
                                  "SilhouetteScore": silhouette_scores})
print(f"\nCluster Selection Metrics:\n{cluster_selection.to_string(index=False)}")
print(f"Best Silhouette k: {cluster_selection.loc[cluster_selection['SilhouetteScore'].idxmax(), 'Clusters']:.0f}")
cluster_selection.to_csv(os.path.join(OUTPUT_DIR, "cluster_selection_metrics.csv"), index=False)

fig = px.line(x=list(k_range), y=inertia, markers=True, title="Elbow Method - Optimal Cluster Selection")
fig.update_layout(template="plotly_dark", xaxis_title="Number of Clusters", yaxis_title="Inertia", title_x=0.5)
save_plotly(fig, "elbow_method")
fig.show()

fig = px.line(x=list(k_range), y=silhouette_scores, markers=True, title="Silhouette Score Analysis")
fig.update_layout(template="plotly_dark", xaxis_title="Clusters", yaxis_title="Silhouette Score", title_x=0.5)
save_plotly(fig, "silhouette_scores")
fig.show()

kmeans = KMeans(n_clusters=4, random_state=RANDOM_STATE, n_init=20)
df_customers["Cluster"] = kmeans.fit_predict(features)

pca = PCA(n_components=2)
components = pca.fit_transform(features)
print(f"\nPCA Explained Variance Ratio (PC1, PC2): {pca.explained_variance_ratio_}")
print(f"PCA Total Variance Explained: {pca.explained_variance_ratio_.sum():.3%}")
pca_df = pd.DataFrame(components, columns=["PC1", "PC2"])
pca_df["Cluster"] = df_customers["Cluster"].astype(str)

fig = px.scatter(pca_df, x="PC1", y="PC2", color="Cluster",
                  title="Customer Segmentation Map", opacity=0.75)
fig.update_traces(marker=dict(size=10))
fig.update_layout(template="plotly_dark", title_x=0.5)
save_plotly(fig, "cluster_pca_map")
fig.show()

# =======================================================
# Phase 9 - Cluster Profiling
# =======================================================
cluster_profile = (
    df_customers.groupby("Cluster")
    [["Recency", "Frequency", "Monetary", "AverageBasketSize", "TotalQuantity"]]
    .mean()
)
print(f"\nCluster Profile:\n{cluster_profile}")
cluster_counts = df_customers["Cluster"].value_counts().sort_index().rename("CustomerCount")
cluster_profile_with_counts = cluster_profile.join(cluster_counts)
print(f"\nCluster Counts and Profiles:\n{cluster_profile_with_counts}")
cluster_profile_with_counts.to_csv(os.path.join(OUTPUT_DIR, "cluster_profiles.csv"))

# Standardised profile heatmap prevents high-unit variables from dominating colour.
profile_z = (cluster_profile - cluster_profile.mean()) / cluster_profile.std(ddof=0)
plt.figure(figsize=(10, 6))
sns.heatmap(profile_z, annot=True, fmt=".2f", center=0, cmap="vlag")
plt.title("Customer Segment Behaviour Profile (Standardised Means)")
save_fig("cluster_profile_heatmap")
plt.show()

# Assign four unique names based on relative cluster profiles. This remains
# robust if numeric K-means labels change between environments.
exceptional_cluster = cluster_profile["Monetary"].idxmax()
lapsed_cluster = cluster_profile.drop(index=exceptional_cluster)["Recency"].idxmax()
remaining_clusters = cluster_profile.drop(index=[exceptional_cluster, lapsed_cluster]).index
loyal_cluster = cluster_profile.loc[remaining_clusters, "Monetary"].idxmax()
occasional_cluster = [c for c in remaining_clusters if c != loyal_cluster][0]
cluster_names = {
    exceptional_cluster: "Exceptional / Wholesale Customers",
    lapsed_cluster: "Lapsed Low-Value Customers",
    loyal_cluster: "Loyal High-Value Customers",
    occasional_cluster: "Occasional Customers",
}

df_customers["ClusterName"] = df_customers["Cluster"].map(cluster_names)
print(f"\nSegment Names:\n{cluster_names}")

# =======================================================
# Phase 10 - Future High-Value Customer Prediction
# =======================================================
future_threshold = model_data["FutureMonetary"].quantile(0.75)
model_data["FutureHighValue"] = (model_data["FutureMonetary"] >= future_threshold).astype(int)
print(f"\nFuture High-Value Threshold (75th percentile): {future_threshold:.2f}")
print(f"Future Class Distribution:\n{model_data['FutureHighValue'].value_counts().sort_index()}")

X_class = model_data[predictor_columns]
y_class = model_data["FutureHighValue"]

X_train, X_test, y_train, y_test = train_test_split(
    X_class, y_class, test_size=0.2, random_state=RANDOM_STATE, stratify=y_class
)

log_model = Pipeline([
    ("scaler", StandardScaler()),
    ("logistic", LogisticRegression(max_iter=2000, class_weight="balanced",
                                     random_state=RANDOM_STATE)),
])
log_model.fit(X_train, y_train)
y_pred = log_model.predict(X_test)

# Logistic Regression models P(HighValue=1) using the sigmoid function:
#   sigmoid(z) = 1 / (1 + e^-z), where z is the weighted sum of features.
# This squashes any real-valued input into a (0,1) probability, which is why
# Logistic Regression (not Linear Regression, whose output is unbounded) is
# the correct tool for a binary classification target.

print("\nLeakage-Safe Future Classification Results")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.3f}")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1:        {f1_score(y_test, y_pred):.3f}")
print(classification_report(y_test, y_pred))

classification_results = pd.DataFrame([{
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    "Recall": recall_score(y_test, y_pred),
    "F1": f1_score(y_test, y_pred),
    "TestCustomers": len(y_test),
}])
classification_results.to_csv(os.path.join(OUTPUT_DIR, "classification_results.csv"), index=False)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 5))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=["Not High-Value", "High-Value"],
            yticklabels=["Not High-Value", "High-Value"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Future High-Value Customer Prediction Matrix")
save_fig("confusion_matrix")
plt.show()

# =======================================================
# Phase 11 - Interpretation
# =======================================================
current_threshold = df_customers["Monetary"].quantile(0.75)
df_customers["CurrentHighValue"] = (df_customers["Monetary"] >= current_threshold).astype(int)
cluster_highvalue = pd.crosstab(df_customers["ClusterName"], df_customers["CurrentHighValue"])
print(f"\nCluster vs Current High Value:\n{cluster_highvalue}")

cluster_revenue = df_customers.groupby("ClusterName")["Monetary"].sum().sort_values()
fig = px.bar(cluster_revenue, orientation="h", title="Revenue Contribution by Customer Segment")
fig.update_layout(template="plotly_dark", title_x=0.5, xaxis_title="Revenue (GBP)",
                  yaxis_title="Segment", showlegend=False)
save_plotly(fig, "revenue_by_segment")
fig.show()

segment_count = df_customers["ClusterName"].value_counts()
fig = px.pie(values=segment_count.values, names=segment_count.index,
             title="Customer Segment Distribution")
fig.update_layout(template="plotly_dark", title_x=0.5)
save_plotly(fig, "segment_distribution")
fig.show()

print("\nFinal Customer Segment Summary")
print(df_customers[["CustomerID", "Cluster", "ClusterName", "Monetary", "CurrentHighValue"]].head(20))

print(f"\nAll report-ready figures saved to: {os.path.abspath(OUTPUT_DIR)}")