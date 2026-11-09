import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set visual style for professional-looking charts
sns.set_theme(style="whitegrid")

# Load Dataset
file_path = r"C:\Users\DELL\OneDrive\Desktop\saad things\Project\Project_for_Elevvo\train.csv"
Data_Set = pd.read_csv(file_path)

# ==========================================
# Phase 1: Data Inspection
# ==========================================
print("--- PHASE 1: DATA INSPECTION ---")
for column in Data_Set.columns:
    null_count = Data_Set[column].isna().sum()
    null_pct = round((null_count / len(Data_Set)) * 100, 2)
    print(f"""{column}: 
          Total Values: {len(Data_Set[column])} | Null Values: {null_count} ({null_pct}%)
          Data Type: {Data_Set[column].dtypes} | Unique Values: {Data_Set[column].nunique()}""")
print("\n" + "="*50 + "\n")


# ==========================================
# Phase 2: Strategic Data Cleaning
# ==========================================
# 1. Fill Embarked with the mode (using [0] to extract the string from the Series)
Data_Set["Embarked"] = Data_Set["Embarked"].fillna(Data_Set["Embarked"].mode()[0])

# 2. Fill Age with median grouped by Pclass AND Sex for higher accuracy
Data_Set["Age"] = Data_Set["Age"].fillna(
    Data_Set.groupby(["Pclass", "Sex"])["Age"].transform("median")
)

# 3. Convert Cabin into a binary indicator ('Has_Cabin') and drop original messy column
Data_Set["Has_Cabin"] = np.where(Data_Set["Cabin"].notna(), "Yes", "No")
Data_Set = Data_Set.drop(columns=["Cabin"])


# ==========================================
# Phase 3: Feature Engineering
# ==========================================
# 1. Create Family_Size and Is_Alone
Data_Set["Family_Size"] = Data_Set["SibSp"] + Data_Set["Parch"] + 1
Data_Set["Is_Alone"] = np.where(Data_Set["Family_Size"] == 1, 1, 0)

# ==========================================
# Phase 4: Summary Statistics & Group-Based Insights
# ==========================================
print("--- PHASE 4: GLOBAL SUMMARY STATISTICS ---")
# Select numeric columns AFTER feature engineering so new columns are included
numeric_df = Data_Set.select_dtypes(include=['number'])

for column in numeric_df.columns:
    col_min = Data_Set[column].min()
    col_max = Data_Set[column].max()
    print(f"""Column: {column}
          Average (Mean):     {round(Data_Set[column].mean(), 2)}
          Median:             {round(Data_Set[column].median(), 2)}
          Minimum:            {col_min}
          Maximum:            {col_max}
          Range:              {round(col_max - col_min, 2)}
          Standard Deviation: {round(Data_Set[column].std(), 3)}
          Variance:           {round(Data_Set[column].var(), 3)}""")
print("\n" + "="*50 + "\n")

# Group-Based Insights
print("--- GROUP-BASED SURVIVAL RATES ---")
gender_insights = Data_Set.groupby("Sex")["Survived"].mean() * 100
class_insights = Data_Set.groupby("Pclass")["Survived"].mean() * 100
embar_insights = Data_Set.groupby("Embarked")["Survived"].mean() * 100
family_insights = Data_Set.groupby("Family_Size")["Survived"].mean() * 100
combined_insights = Data_Set.groupby(["Pclass", "Sex"])["Survived"].mean() * 100

print("Survival Rate by Gender (%):\n", round(gender_insights, 2), "\n")
print("Survival Rate by Passenger Class (%):\n", round(class_insights, 2), "\n")
print("Survival Rate by Embarkation Port (%):\n", round(embar_insights, 2), "\n")
print("Survival Rate by Family Size (%):\n", round(family_insights, 2), "\n")
print("Survival Rate by Class & Gender (%):\n", round(combined_insights, 2), "\n")


# ==========================================
# Phase 5: Visualizations & Bonus Tasks
# ==========================================

# 1. Bonus Bar Plots (1 Row, 3 Columns)
fig, axis = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Survival Rate by Class
sns.barplot(data=Data_Set, x="Pclass", y="Survived", hue="Pclass", palette="muted", legend=False, ax=axis[0], errorbar=None)
axis[0].set_title("Survival Rate by Passenger Class", fontsize=13, fontweight="bold")
axis[0].set_xlabel("Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)")
axis[0].set_ylabel("Survival Probability")

# Plot 2: Survival Rate by Class and Gender
sns.barplot(data=Data_Set, x="Pclass", y="Survived", hue="Sex", palette="muted", ax=axis[1], errorbar=None)
axis[1].set_title("Survival Rate by Class & Gender", fontsize=13, fontweight="bold")
axis[1].set_xlabel("Passenger Class")
axis[1].set_ylabel("Survival Probability")

# Plot 3: Survival Rate by Family Size
sns.barplot(data=Data_Set, x="Family_Size", y="Survived", hue="Is_Alone", palette="muted", dodge=False, ax=axis[2], errorbar=None)
axis[2].set_title("Survival Rate by Family Size", fontsize=13, fontweight="bold")
axis[2].set_xlabel("Total Family Size")
axis[2].set_ylabel("Survival Probability")

plt.tight_layout()
plt.show()


# 2. Continuous Data Plot: Age Distribution vs. Survival
plt.figure(figsize=(10, 5))
sns.kdeplot(data=Data_Set, x="Age", hue="Survived", fill=True, palette="crest", common_norm=False, alpha=0.5)
plt.title("Age Distribution: Survivors (1) vs. Non-Survivors (0)", fontsize=14, fontweight="bold")
plt.xlabel("Passenger Age")
plt.ylabel("Density")
plt.show()


# 3. Bonus Heatmap: Feature Correlations
# Convert key categorical variables to numeric so they appear in the correlation matrix
corr_df = Data_Set.copy()
corr_df["Sex_Num"] = corr_df["Sex"].map({"male": 0, "female": 1})
corr_df["Has_Cabin_Num"] = corr_df["Has_Cabin"].map({"No": 0, "Yes": 1})
corr_df["Embarked_Num"] = corr_df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

# Select relevant numeric columns for correlation (excluding random ID)
cols_to_correlate = ["Survived", "Pclass", "Sex_Num", "Age", "SibSp", "Parch", "Fare", "Family_Size", "Is_Alone", "Has_Cabin_Num", "Embarked_Num"]
correlation_matrix = corr_df[cols_to_correlate].corr()

plt.figure(figsize=(11, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", linewidths=0.5, vmin=-1, vmax=1)
plt.title("Titanic Feature Correlation Heatmap", fontsize=15, fontweight="bold")
plt.tight_layout()
plt.show()


# ==========================================
# Phase 6: Executive Summary
# ==========================================
print("""
===================================================
EXECUTIVE SUMMARY: KEY TAKEAWAYS FROM TITANIC EDA
===================================================
1. Gender Disparity: Females had a ~74.2% survival rate compared to just ~18.9% for males, confirming the 'women and children first' protocol.
2. Socioeconomic Status (Class & Cabin): 1st-class passengers had a ~63.0% survival rate, whereas 3rd-class passengers had only a ~24.2% survival rate. Having a recorded cabin strongly correlated with survival.
3. Family Dynamics: Passengers traveling in small families (2 to 4 total members) had the highest survival rates. Traveling completely alone or in very large families (5+) significantly reduced survival odds.
4. Age Factor: The age density plot reveals a sharp survival spike for infants and young children (ages 0-5), regardless of class.
""")