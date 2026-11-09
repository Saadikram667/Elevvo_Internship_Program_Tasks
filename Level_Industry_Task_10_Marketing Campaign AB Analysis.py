import logging
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from scipy import stats

# Set up standard logging for data pipelines
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class MarketingABTester:
    """
    Evaluates A/B test results to determine Conversion Rate Optimization (CRO) success.
    Defaults to a 95% confidence level (alpha = 0.05).
    """
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.metrics: Dict[str, Any] = {}

    def load_data(self, filepath: str) -> pd.DataFrame:
        """Loads and validates the Kaggle Marketing A/B Testing Dataset."""
        try:
            df = pd.read_csv(filepath)
            # Standardize column names for easier access
            df.columns = df.columns.str.lower().str.replace(' ', '_')
            logging.info(f"Loaded dataset with {len(df):,} records.")
            return df
        except FileNotFoundError:
            logging.warning(f"File {filepath} not found. Generating sample data for demonstration.")
            return self._generate_mock_data()

    def _generate_mock_data(self) -> pd.DataFrame:
        """Generates realistic mock data if the Kaggle CSV is missing."""
        np.random.seed(42)
        n_size = 50000
        # Control group sees PSA (Public Service Announcement), Test group sees new Ad
        groups = np.random.choice(['psa', 'ad'], size=n_size, p=[0.1, 0.9])
        
        # Simulate conversion rates: ~1.5% for PSA, ~2.5% for Ad
        conversions = np.zeros(n_size, dtype=bool)
        conversions[groups == 'psa'] = np.random.binomial(1, 0.015, sum(groups == 'psa'))
        conversions[groups == 'ad'] = np.random.binomial(1, 0.025, sum(groups == 'ad'))
        
        return pd.DataFrame({'user_id': range(n_size), 'test_group': groups, 'converted': conversions})

    def analyze_campaign(self, df: pd.DataFrame, control_label: str = 'psa', test_label: str = 'ad') -> None:
        """Calculates conversion rates, runs the Z-test, and computes the business lift."""
        if 'test_group' not in df.columns or 'converted' not in df.columns:
            raise ValueError("Dataset missing required columns: 'test_group' or 'converted'")

        # 1. Core Task: Calculate base metrics
        summary = df.groupby('test_group')['converted'].agg(['count', 'sum'])
        summary.rename(columns={'count': 'total_users', 'sum': 'conversions'}, inplace=True)
        summary['conversion_rate'] = summary['conversions'] / summary['total_users']
        
        control = summary.loc[control_label]
        test = summary.loc[test_label]
        
        # 2. Industry Constraint: Statistical Significance (2-Proportion Z-Test)
        successes = np.array([control['conversions'], test['conversions']])
        nobs = np.array([control['total_users'], test['total_users']])
        
        z_stat, p_value = proportions_ztest(count=successes, nobs=nobs, alternative='two-sided')
        
        # Calculate Confidence Intervals for the conversion rates
        ci_low, ci_upp = proportion_confint(count=successes, nobs=nobs, alpha=self.alpha, method='normal')
        
        # 3. Bonus: Calculate Lift and Projected Business Impact
        relative_lift = (test['conversion_rate'] - control['conversion_rate']) / control['conversion_rate']
        absolute_lift = test['conversion_rate'] - control['conversion_rate']
        
        total_audience = df['user_id'].nunique()
        projected_additional_customers = int(total_audience * absolute_lift)

        self.metrics = {
            'control_cr': control['conversion_rate'],
            'test_cr': test['conversion_rate'],
            'control_ci': (ci_low[0], ci_upp[0]),
            'test_ci': (ci_low[1], ci_upp[1]),
            'z_statistic': z_stat,
            'p_value': p_value,
            'is_significant': p_value < self.alpha,
            'relative_lift': relative_lift,
            'projected_new_customers': projected_additional_customers
        }
        
        self._print_executive_summary()

    def _print_executive_summary(self) -> None:
        """Outputs a clean, professional summary of the statistical inference."""
        m = self.metrics
        print("\n" + "="*50)
        print(" A/B TEST RESULTS: EXECUTIVE SUMMARY")
        print("="*50)
        print(f"Control Conversion Rate: {m['control_cr']:.2%} (95% CI: {m['control_ci'][0]:.2%} - {m['control_ci'][1]:.2%})")
        print(f"Test Conversion Rate:    {m['test_cr']:.2%} (95% CI: {m['test_ci'][0]:.2%} - {m['test_ci'][1]:.2%})")
        print(f"\nStatistical Significance (alpha=0.05): {'YES' if m['is_significant'] else 'NO'}")
        print(f"P-Value: {m['p_value']:.5f} | Z-Statistic: {m['z_statistic']:.3f}")
        print(f"\nBusiness Impact (Bonus):")
        print(f"Relative Lift: +{m['relative_lift']:.2%}")
        print(f"Projected additional customers at 100% rollout: {m['projected_new_customers']:,}")
        print("="*50 + "\n")

    def visualize_lift(self) -> None:
        """Bonus: Visualizes the improvement (Lift) with confidence intervals."""
        if not self.metrics:
            logging.error("No metrics to plot. Run analyze_campaign() first.")
            return

        m = self.metrics
        labels = ['Control Group (PSA)', 'Test Group (Ad)']
        rates = [m['control_cr'], m['test_cr']]
        
        # Calculate error margins for the plot
        error_control = m['control_ci'][1] - m['control_cr']
        error_test = m['test_ci'][1] - m['test_cr']
        errors = [error_control, error_test]

        sns.set_theme(style="whitegrid")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(labels, rates, yerr=errors, capsize=10, 
                      color=['#95a5a6', '#2ecc71'], alpha=0.8, edgecolor='black')

        ax.set_title("Marketing Campaign Conversion Rates\n(with 95% Confidence Intervals)", 
                     fontsize=14, weight='bold', pad=15)
        ax.set_ylabel("Conversion Rate", weight='bold')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))

        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{height:.2%}', ha='center', va='center', 
                    color='black', weight='bold', fontsize=12)

        # Highlight statistical significance
        sig_text = f"Statistically Significant Lift: +{m['relative_lift']:.1%}" if m['is_significant'] else "No Significant Lift Detected"
        props = dict(boxstyle='round', facecolor='#f1c40f' if m['is_significant'] else '#e74c3c', alpha=0.3)
        ax.text(0.5, 0.95, sig_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', horizontalalignment='center', bbox=props, weight='bold')

        sns.despine()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize the tester (industry constraint: >95% confidence means alpha=0.05)
    ab_tester = MarketingABTester(alpha=0.05)
    
    # Load Kaggle Marketing A/B Testing Dataset
    campaign_data = ab_tester.load_data('marketing_AB.csv')
    
    # Run the core task and bonus projections
    ab_tester.analyze_campaign(campaign_data)
    
    # Render the bonus visualization
    ab_tester.visualize_lift()