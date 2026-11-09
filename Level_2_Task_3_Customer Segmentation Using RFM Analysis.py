import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style
sns.set_theme(style="whitegrid")

# ==========================================
# Phase 1: Load & Clean Data
# ==========================================
# Note: Update the file path to your local CSV or Excel file (.read_excel if .xlsx)
file_path = r"C:\Users\DELL\OneDrive\Desktop\saad things\Project\Project_for_Elevvo\Online_Retail.csv"
df = pd.read_csv(file_path)

print("--- PHASE 1: DATA CLEANING ---")

# 1. Drop rows without a CustomerID
df = df.dropna(subset=["CustomerID"])

# 2. Remove cancelled orders (Quantity <= 0) and invalid prices (UnitPrice <= 0)
df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]

# 3. Convert CustomerID to integer string and InvoiceDate to datetime
df["CustomerID"] = df["CustomerID"].astype(int).astype(str)
df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])

# 4. Calculate total spend per line item
df["Total_Spend"] = df["Quantity"] * df["UnitPrice"]

print(f"Cleaned Rows: {len(df)} | Unique Customers: {df['CustomerID'].nunique()}\n")


# ==========================================
# Phase 2: Feature Engineering (Calculate R, F, M)
# ==========================================
# Set reference snapshot date as 1 day after the last recorded purchase
snapshot_date = df["InvoiceDate"].max() + dt.timedelta(days=1)

# Aggregate data by CustomerID
rfm = df.groupby("CustomerID").agg({
    "InvoiceDate": lambda x: (snapshot_date - x.max()).days, # Recency
    "InvoiceNo": "nunique",                                  # Frequency (unique orders)
    "Total_Spend": "sum"                                     # Monetary
}).reset_index()

# Rename columns
rfm.rename(columns={
    "InvoiceDate": "Recency",
    "InvoiceNo": "Frequency",
    "Total_Spend": "Monetary"
}, inplace=True)


# ==========================================
# Phase 3: RFM Scoring (1 to 5 Quintiles)
# ==========================================
# Recency: Lower days = Higher score (5 is most recent, 1 is least recent)
rfm["R_Score"] = pd.qcut(rfm["Recency"], q=5, labels=[5, 4, 3, 2, 1]).astype(int)

# Frequency: Use .rank(method='first') to handle ties in purchase counts cleanly
rfm["F_Score"] = pd.qcut(rfm["Frequency"].rank(method="first"), q=5, labels=[1, 2, 3, 4, 5]).astype(int)

# Monetary: Higher spend = Higher score (5 is highest spend, 1 is lowest)
rfm["M_Score"] = pd.qcut(rfm["Monetary"], q=5, labels=[1, 2, 3, 4, 5]).astype(int)

# Combine into an RFM_Score string and calculate overall sum
rfm["RFM_Code"] = rfm["R_Score"].astype(str) + rfm["F_Score"].astype(str) + rfm["M_Score"].astype(str)
rfm["RFM_Sum"] = rfm["R_Score"] + rfm["F_Score"] + rfm["M_Score"]


# ==========================================
# Phase 4: Segmentation Logic
# ==========================================
def assign_segment(row):
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif r >= 3 and f >= 4:
        return "Loyal Customers"
    elif r >= 4 and f <= 2:
        return "New / Promising"
    elif r == 3 and f <= 3:
        return "Need Attention"
    elif r <= 2 and f >= 3:
        return "At Risk"
    else:
        return "Inactive / Hibernating"

rfm["Segment"] = rfm.apply(assign_segment, axis=1)

# Summary table of segments
segment_summary = rfm.groupby("Segment").agg(
    Customer_Count=("CustomerID", "count"),
    Avg_Recency_Days=("Recency", "mean"),
    Avg_Frequency=("Frequency", "mean"),
    Avg_Monetary_Spend=("Monetary", "mean")
).round(2).sort_values(by="Customer_Count", ascending=False)

print("--- PHASE 4: RFM SEGMENT SUMMARY ---")
print(segment_summary, "\n")


# ==========================================
# Phase 5: Bonus Visualizations (Bar Charts & Heatmap)
# ==========================================
fig, axis = plt.subplots(1, 2, figsize=(16, 6))

# Bonus Plot 1: Bar Chart of Customer Count by Segment
sns.countplot(
    data=rfm,
    y="Segment",
    order=rfm["Segment"].value_counts().index,
    hue="Segment",
    palette="viridis",
    legend=False,
    ax=axis[0]
)
axis[0].set_title("Customer Distribution Across RFM Segments", fontsize=14, fontweight="bold")
axis[0].set_xlabel("Number of Customers")
axis[0].set_ylabel("Customer Segment")

# Bonus Plot 2: Bar Chart of Average Monetary Value by Segment
sns.barplot(
    data=rfm,
    y="Segment",
    x="Monetary",
    order=rfm["Segment"].value_counts().index,
    hue="Segment",
    palette="magma",
    legend=False,
    errorbar=None,
    ax=axis[1]
)
axis[1].set_title("Average Spend (Monetary) by Segment", fontsize=14, fontweight="bold")
axis[1].set_xlabel("Average Spend ($)")
axis[1].set_ylabel("")

plt.tight_layout()
plt.show()

# Bonus Plot 3: Heatmap of Average Monetary Spend across R and F Scores
rfm_pivot = rfm.pivot_table(index="R_Score", columns="F_Score", values="Monetary", aggfunc="mean").round(0)
# Sort index so R_Score=5 is at the top of the heatmap
rfm_pivot = rfm_pivot.sort_index(ascending=False)

plt.figure(figsize=(9, 6))
sns.heatmap(rfm_pivot, annot=True, fmt=".0f", cmap="YlGnBu", linewidths=0.5)
plt.title("RFM Heatmap: Average Spend ($) by Recency & Frequency Scores", fontsize=14, fontweight="bold")
plt.xlabel("Frequency Score (1 = Lowest, 5 = Highest)")
plt.ylabel("Recency Score (1 = Oldest, 5 = Most Recent)")
plt.tight_layout()
plt.show()


# ==========================================
# Phase 6: Marketing Ideas by Segment
# ==========================================
print("""
=====================================================================
MARKETING RECOMMENDATIONS BY CUSTOMER SEGMENT
=====================================================================
1. Champions (High R, High F, High M):
   - Idea: VIP loyalty perks, early access to new products, and referral bonuses. Avoid heavy discounting since they already buy at full price.
2. Loyal Customers (Recent & Frequent):
   - Idea: Volume discounts, bundle offers, or tiered loyalty points to increase their average basket size (Monetary value).
3. New / Promising (High R, Low F):
   - Idea: Welcome onboarding emails and a time-sensitive discount coupon on their second order to build a repeat purchasing habit.
4. Need Attention (Moderate R & F):
   - Idea: Personalized product recommendations based on past purchases and limited-time promotional offers.
5. At Risk (Low R, High F - Used to buy often, haven't visited lately):
   - Idea: Urgent "We Miss You!" win-back campaigns, satisfaction surveys, and steep discounts to prevent churn.
6. Inactive / Hibernating (Low R, Low F):
   - Idea: Automated re-engagement emails during major seasonal sales; avoid spending heavy paid-ad budget on this group.
""")
