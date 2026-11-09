import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

# load data (update filename to match the specific year you downloaded)
df = pd.read_csv('kaggle_survey_2021_responses.csv', low_memory=False)

# 1. Data Cleaning
# kaggle surveys usually have the question text in the first row, so we drop it
if 'What is your age' in str(df.iloc[0].values):
    df = df.iloc[1:].reset_index(drop=True)

# drop exact duplicates
df.drop_duplicates(inplace=True)

# drop columns that are mostly empty (>50% missing data)
thresh = len(df) * 0.5
df.dropna(thresh=thresh, axis=1, inplace=True)

# fill remaining NAs: median for numerical, 'Unknown' for text
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna('Unknown', inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# 2. Categorical Handling
# manual mapping for ordinal data like education (assuming Q4 is education level)
edu_map = {
    'No formal education past high school': 0,
    'Some college/university study without earning a bachelor’s degree': 1,
    'Bachelor’s degree': 2,
    'Master’s degree': 3,
    'Doctoral degree': 4,
    'Professional doctorate': 5,
    'I prefer not to answer': -1,
    'Unknown': -1
}
if 'Q4' in df.columns:
    df['education_num'] = df['Q4'].map(edu_map)

# label encoding for nominal data like country (Q3)
le = LabelEncoder()
if 'Q3' in df.columns:
    df['country_encoded'] = le.fit_transform(df['Q3'].astype(str))

# 3. Insights & Bonus Charting (Summary Dashboard)
sns.set_theme(style="darkgrid")
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Kaggle Survey - Top 5 Insights Dashboard', fontsize=16)

# plot 1: top 5 age ranges (Q1)
if 'Q1' in df.columns:
    top_ages = df['Q1'].value_counts().head(5)
    sns.barplot(x=top_ages.values, y=top_ages.index, ax=axes[0,0], palette='viridis')
    axes[0,0].set_title('Top 5 Age Demographics')

# plot 2: top 5 education levels (Q4)
if 'Q4' in df.columns:
    top_edu = df['Q4'].value_counts().drop('Unknown', errors='ignore').head(5)
    sns.barplot(x=top_edu.values, y=top_edu.index, ax=axes[0,1], palette='magma')
    axes[0,1].set_title('Top 5 Education Levels')

# plot 3: top 5 current roles (Q5)
if 'Q5' in df.columns:
    top_roles = df['Q5'].value_counts().drop('Unknown', errors='ignore').head(5)
    sns.barplot(x=top_roles.values, y=top_roles.index, ax=axes[1,0], palette='crest')
    axes[1,0].set_title('Top 5 Job Titles')

# plot 4: top 5 coding experience levels (Q6)
if 'Q6' in df.columns:
    top_exp = df['Q6'].value_counts().drop('Unknown', errors='ignore').head(5)
    sns.barplot(x=top_exp.values, y=top_exp.index, ax=axes[1,1], palette='rocket')
    axes[1,1].set_title('Top 5 Programming Experience Levels')

plt.tight_layout()
plt.show()