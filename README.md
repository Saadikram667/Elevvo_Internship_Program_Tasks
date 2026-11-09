**Repository Description (Short - for the GitHub "About" section)**
A comprehensive data analytics portfolio demonstrating exploratory data analysis (EDA), RFM customer segmentation, web scraping, time series forecasting, and statistical A/B testing using Python.

---

**README.md File (Full - for the repository homepage)**

# Data Analytics & Science Portfolio

This repository contains a series of data analytics projects completed to demonstrate end-to-end data processing, statistical analysis, and visualization skills. The projects range from data cleaning and exploratory data analysis (EDA) to advanced techniques like time series forecasting and A/B testing.

## 🛠️ Tools & Technologies

* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Statistical Analysis:** Statsmodels, SciPy
* **Web Scraping:** BeautifulSoup, Requests
* **Machine Learning / Preprocessing:** Scikit-Learn

## 📂 Project Breakdown

### 1. Exploratory Data Analysis (EDA) on Titanic Dataset



* **File:** `Level_1_Task_2_Exploratory Data Analysis on Titanic Dataset.py`


* **Description:** Cleaned and analyzed the classic Titanic dataset. Handled missing values (using grouped medians and modes), engineered new features (Family Size, Cabin Indicator), and generated summary statistics.


* **Key Visuals:** Survival rate bar plots by class and gender, age distribution density plots, and a feature correlation heatmap.



### 2. Customer Segmentation Using RFM Analysis



* **File:** `Level_2_Task_3_Customer Segmentation Using RFM Analysis.py`


* **Description:** Processed online retail data to calculate Recency, Frequency, and Monetary (RFM) scores. Grouped customers into actionable segments (e.g., Champions, Loyal Customers, At Risk) to inform targeted marketing strategies.


* **Key Visuals:** Segment distribution bar charts, average spend by segment, and an RFM heatmap.



### 3. Data Cleaning and Insight Generation from Survey Data



* **File:** `Level_2_Task_4_Data Cleaning and Insight Generation from Survey Data.py`


* **Description:** Handled messy Kaggle survey data by dropping empty columns, imputing missing values, and applying label encoding for categorical variables.


* **Key Visuals:** A multi-plot summary dashboard displaying the top 5 age demographics, education levels, job titles, and programming experience levels.



### 4. Web Scraping and Analysis of Job Postings



* **File:** `Level_3_Task_6_Web Scraping and Analysis of Job Postings.py`


* **Description:** Built a polite, `robots.txt`-compliant web scraper using BeautifulSoup to extract job titles, companies, locations, and posted dates. Simulated keyword matching to extract required technical skills.


* **Key Visuals:** A count plot showcasing the most in-demand technical skills across top job market regions.



### 5. Time Series Breakdown of Retail Sales



* **File:** `Level_3_task_7_Time Series Breakdown of Retail Sales.py`


* **Description:** Resampled weekly retail data into monthly revenues and applied moving averages to identify trends. Utilized the Holt-Winters Exponential Smoothing model (capturing trend and seasonality) to forecast sales 6 months into the future.


* **Key Visuals:** Combined line chart featuring historical trends, a 3-month moving average, predictive forecasting, and a regional revenue breakdown.



### 6. Marketing Campaign A/B Analysis



* **File:** `Level_Industry_Task_10_Marketing Campaign AB Analysis.py`


* **Description:** Evaluated the success of a Conversion Rate Optimization (CRO) campaign. Calculated conversion rates for Control (PSA) and Test (Ad) groups, and performed a 2-Proportion Z-Test to ensure statistical significance at a 95% confidence level. Projected the business impact of a 100% rollout.


* **Key Visuals:** Bar charts displaying conversion rates with calculated error margins (95% Confidence Intervals).



## 🚀 How to Run

1. Clone this repository: `git clone [your-repository-url]`
2. Ensure you have the required libraries installed: `pip install pandas numpy matplotlib seaborn scipy statsmodels scikit-learn beautifulsoup4 requests`
3. Download the respective datasets (Titanic, Online Retail, Kaggle Survey, Walmart Sales, Marketing A/B) and update the local file paths in the scripts.
4. Run individual `.py` files to view the data pipelines and visualizations.

## 🎓 Certifications

This portfolio is published in conjunction with the successful completion of the **Data Science Essentials with Python** certification provided by the Cisco Networking Academy.
