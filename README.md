# Retail Customer Analytics: Sales Prediction, Customer Segmentation and Business Insights

## 📊 Project Overview

This project applies **data analytics and machine-learning techniques** to the UCI Online Retail dataset to understand customer purchasing behaviour, identify meaningful customer segments, and predict future customer value.

The analysis is designed around a UK-based online giftware retailer and uses transaction-level purchasing data covering the period from **1 December 2010 to 9 December 2011**.

The project combines:

* Exploratory Data Analysis (EDA)
* Data cleaning and preprocessing
* RFM-style customer analysis
* Customer segmentation using **K-Means clustering**
* Future customer monetary-value prediction using **Linear Regression** and **Random Forest Regression**
* Future high-value customer classification using **Logistic Regression**
* Business-oriented interpretation and recommendations

---

## 🎯 Research Questions

The analysis addresses three main research questions:

1. **Can customers be separated into coherent and commercially interpretable behavioural segments?**

2. **Which behavioural characteristics distinguish loyal, occasional, lapsed and exceptional high-volume customers?**

3. **How accurately can earlier customer behaviour predict future monetary value and future high-value status?**

---

## 📁 Dataset

The project uses the **UCI Online Retail dataset**.

The original dataset contains transaction-level information including:

| Variable      | Description                  |
| ------------- | ---------------------------- |
| `InvoiceNo`   | Invoice/transaction number   |
| `StockCode`   | Product code                 |
| `Description` | Product description          |
| `Quantity`    | Quantity purchased           |
| `InvoiceDate` | Date and time of transaction |
| `UnitPrice`   | Price per unit               |
| `CustomerID`  | Customer identifier          |
| `Country`     | Customer's country           |

The analysis begins with **541,909 invoice-line records**.

After data cleaning and validation, **392,692 transaction rows** were retained for analysis, representing approximately **72.46%** of the original dataset.

The customer-level analytical dataset contains **4,338 customers**.

---

## 🧹 Data Preparation

The preprocessing stage addresses several data-quality issues:

* Missing `CustomerID` values
* Duplicate records
* Cancelled invoices
* Non-positive quantities
* Invalid/non-positive unit prices
* Date conversion and validation

A transaction-level monetary variable was created:

```python
TotalPrice = Quantity × UnitPrice
```

Customer-level features were then engineered, including:

* Recency
* Frequency
* Monetary value
* Total quantity purchased
* Average basket size
* UK customer indicator

For clustering, skewed monetary and frequency variables were transformed using `log1p`, followed by standardisation.

---

## 🔎 Exploratory Data Analysis

The project investigates:

* Daily revenue patterns
* Top-selling products
* Revenue by country
* Customer monetary-value distributions
* Purchase-frequency distributions
* Correlations between behavioural variables
* Outliers in customer-level purchasing behaviour

The analysis also examines the relationship between customer frequency, monetary value, quantity and recency.

---

# 🤖 Machine Learning Methodology

## 1. Customer Segmentation — K-Means

Customer segmentation is performed using **K-Means clustering**.

The clustering variables are:

* Recency
* Log-transformed Frequency
* Log-transformed Monetary Value
* Total Quantity

Candidate cluster solutions from **k = 2 to k = 10** are evaluated using:

* Within-cluster inertia
* Silhouette score

K-Means uses:

```python
n_init = 20
random_state = 42
```

The final analysis uses **4 customer segments**.

### Identified Customer Segments

| Segment                 | Customers | Description                                                                    |
| ----------------------- | --------: | ------------------------------------------------------------------------------ |
| Occasional              |     2,050 | Infrequent customers with relatively recent but limited purchasing             |
| Loyal high-value        |     1,293 | Frequent customers generating substantial monetary value                       |
| Lapsed low-value        |       981 | Customers with long periods since their last purchase and relatively low value |
| Exceptional high-volume |        14 | Very high-frequency and exceptionally high-value customers                     |

Principal Component Analysis (PCA) is used only to visualise the resulting clusters in two dimensions.

---

## 2. Future Monetary-Value Prediction

The project predicts customer spending during a future period.

A temporal cutoff of:

**1 September 2011**

is used.

Historical customer behaviour before the cutoff forms the predictor variables, while spending from the cutoff onwards forms the future monetary-value target.

This creates a leakage-safe temporal modelling framework.

### Models

Two regression models are evaluated:

1. **Linear Regression**
2. **Random Forest Regression**

The predictors include:

* Historical recency
* Historical frequency
* Historical monetary value
* Historical total quantity
* Historical average basket size
* UK customer indicator

### Evaluation Metrics

The regression models are evaluated using:

* Mean Absolute Error (MAE)
* Mean Squared Error (MSE)
* Root Mean Squared Error (RMSE)
* R²

Both **80/20** and **70/30** train-test splits are examined as a sensitivity analysis.

---

## 3. Future High-Value Customer Classification

Customers are classified according to whether their future spending reaches the **75th percentile threshold**.

The future spending threshold used in the analysis is:

**£747.10**

Customers above this threshold are classified as future high-value customers.

A **Logistic Regression** classifier is used with:

* Standardised predictors
* Stratified train-test splitting
* Balanced class weights

### Classification Metrics

Performance is assessed using:

* Accuracy
* Precision
* Recall
* F1-score
* Specificity
* Balanced accuracy
* Confusion matrix

---

# 📈 Key Results

## Customer Segmentation

The four-cluster solution produced a silhouette score of approximately:

**0.410**

The largest customer segment is the **Occasional** group, while the **Exceptional high-volume** segment contains only 14 customers but contributes a substantial proportion of revenue.

The Loyal high-value segment contains approximately **29.81% of customers** and accounts for approximately **61.36% of revenue** in the customer-level segmentation analysis.

---

## Future Monetary-Value Regression

The regression results show that historical monetary behaviour is strongly associated with future customer spending.

For the 80/20 split:

| Model             |     MAE |      RMSE |    R² |
| ----------------- | ------: | --------: | ----: |
| Linear Regression | £857.36 | £3,958.25 | 0.659 |
| Random Forest     | £970.26 | £4,929.67 | 0.471 |

A 70/30 split was also evaluated to assess sensitivity to the train-test allocation.

The Random Forest feature-importance analysis indicates that historical monetary value is the most influential predictor, followed by historical total quantity and average basket size.

---

## Future High-Value Classification

The Logistic Regression classifier achieved:

| Metric            | Result |
| ----------------- | -----: |
| Accuracy          |  0.810 |
| Precision         |  0.608 |
| Recall            |  0.681 |
| F1-score          |  0.642 |
| Specificity       |  0.853 |
| Balanced Accuracy |  0.767 |

The confusion matrix contained:

* 425 true negatives
* 113 true positives
* 73 false positives
* 53 false negatives

---

# 💼 Business Insights

The analysis provides several potential business applications.

### 1. Retain Loyal High-Value Customers

The loyal segment represents a substantial proportion of customer revenue. These customers can be prioritised for retention initiatives and targeted engagement.

### 2. Develop Occasional Customers

The occasional segment represents the largest customer group. Increasing purchase frequency among these customers could provide an opportunity for future value growth.

### 3. Selective Lapsed-Customer Re-engagement

Lapsed customers can be considered for targeted win-back campaigns, while avoiding unnecessary expenditure across the entire lapsed population.

### 4. Manage Exceptional Accounts Individually

The exceptional high-volume segment contains only a small number of customers but generates a disproportionately large amount of revenue. These accounts may therefore benefit from closer account-level management.

### 5. Use Predictive Models for Customer Prioritisation

The classification model can help identify customers with a higher probability of becoming future high-value customers.

The regression model can provide an estimate of future monetary value that can support broader revenue planning and budgeting.

---

# ⚠️ Limitations

Several limitations should be considered when interpreting the results.

### Customer Identification

Customers without a valid `CustomerID` are excluded from customer-level modelling.

### Profitability

The analysis focuses on purchasing value rather than true profitability. Product costs, margins and other operating costs are not available.

### Marketing Information

The dataset does not provide information about marketing exposure, campaigns or customer demographics.

### Time Period

The dataset covers approximately one year and therefore provides limited evidence about longer-term customer behaviour.

### Heavy-Tailed Future Spending

Future spending is zero-inflated and highly skewed. Extreme-value customers create challenges for regression models.

### Model Limitations

The regression residual analysis indicates heteroscedasticity and extreme residuals, while the Random Forest tends to underpredict extreme future spending values.

---

# 🔮 Future Work

Potential extensions include:

* Temporal cross-validation
* Two-stage models for zero versus positive future spending
* Log-transformed target modelling
* Gamma or Tweedie regression
* Robust regression approaches
* ROC-AUC and precision-recall analysis
* Probability calibration
* More extensive clustering-stability analysis
* Incorporating product-level and margin information
* Longer observation periods where available

---

# 🛠️ Technologies Used

The project was developed in Python using:

* **Python**
* **Pandas**
* **NumPy**
* **Matplotlib**
* **Seaborn**
* **Scikit-learn**
* **OpenPyXL**

Machine-learning methods include:

* K-Means Clustering
* Principal Component Analysis
* Linear Regression
* Random Forest Regression
* Logistic Regression
* StandardScaler
* Train/Test Splitting

---

# 📂 Project Structure

```text
Retail-Customer-Analytics/
│
├── FDA_customer_analytics_revised.py
├── Retail_Customer_Analytics_Report.pdf
├── README.md
│
├── data/
│   └── online_retail.xlsx
│
└── outputs/
    ├── figures/
    └── tables/
```

> The dataset may need to be downloaded separately depending on repository licensing and file-size considerations.

---

# ▶️ How to Run the Project

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd Retail-Customer-Analytics
```

## 2. Install the required libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

## 3. Add the dataset

Place the UCI Online Retail Excel dataset in the appropriate project directory.

Update the `DATA_PATH` variable in the Python script if necessary.

## 4. Run the Python script

```bash
python FDA_customer_analytics_revised.py
```

The script performs:

1. Data loading
2. Data-quality assessment
3. Cleaning
4. Feature engineering
5. Exploratory analysis
6. Customer segmentation
7. Future-value regression
8. High-value customer classification
9. Visualisation
10. Output generation

---

# 🔁 Reproducibility

A fixed random seed of:

```python
random_state = 42
```

is used in the machine-learning workflow where applicable.

The project is structured so that the main analytical workflow can be reproduced by running the Python script with the required dataset and dependencies.

---

# 📚 Dataset Source

**UCI Machine Learning Repository — Online Retail Dataset**

Dataset:

> Online Retail

The dataset contains transactions from a UK-based online retailer and is widely used for customer analytics and machine-learning applications.

---

# 👩‍💻 Author

**Trisha Kampani**

MSc Fundamentals of Data Analytics

Project: **Retail Customer Analytics: Sales Prediction, Customer Segmentation and Business Insights**

---

## 📌 Academic Note

This repository contains the code and supporting materials for an academic data analytics project. The analysis is intended for educational and research purposes and should not be interpreted as a production customer-scoring system without further validation.
**
