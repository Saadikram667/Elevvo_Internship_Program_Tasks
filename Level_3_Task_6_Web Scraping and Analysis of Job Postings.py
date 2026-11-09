import logging
import time
from typing import List, Dict, Optional
from urllib.robotparser import RobotFileParser
import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configure professional logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class JobMarketAnalyzer:
    """Scrapes and analyzes job market data with rate limiting and robots.txt compliance."""
    
    def __init__(self, base_url: str, target_url: str):
        self.base_url = base_url
        self.target_url = target_url
        self.session = requests.Session()
        # Using a custom user-agent is standard practice for ethical scraping
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Data Science Portfolio Project; Research Purposes)'
        })

    def _is_scraping_allowed(self) -> bool:
        """Parses robots.txt to ensure compliance."""
        rp = RobotFileParser()
        rp.set_url(f"{self.base_url}/robots.txt")
        try:
            rp.read()
            return rp.can_fetch(self.session.headers['User-Agent'], self.target_url)
        except Exception as e:
            logging.warning(f"Could not parse robots.txt: {e}. Defaulting to polite scraping.")
            return True

    def extract_jobs(self) -> pd.DataFrame:
        """Extracts job titles, companies, locations, and skills into a DataFrame."""
        if not self._is_scraping_allowed():
            logging.error("Scraping blocked by robots.txt policy.")
            return pd.DataFrame()

        logging.info(f"Fetching data from {self.target_url}")
        try:
            response = self.session.get(self.target_url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch page: {e}")
            return pd.DataFrame()

        soup = BeautifulSoup(response.content, "html.parser")
        jobs_data: List[Dict[str, str]] = []
        
        for card in soup.find_all("div", class_="card-content"):
            title = card.find("h2", class_="title")
            # Using safe extraction to avoid NoneType AttributeErrors
            if not title:
                continue
                
            title_text = title.text.strip()
            company = card.find("h3", class_="company").text.strip() if card.find("h3") else "Unknown"
            location = card.find("p", class_="location").text.strip() if card.find("p", class_="location") else "Unknown"
            date_elem = card.find("time")
            date_posted = date_elem["datetime"] if date_elem else pd.Timestamp.now().strftime('%Y-%m-%d')
            
            # NLP-based skill extraction mock (simulating keyword matching)
            title_lower = title_text.lower()
            skills = []
            if "python" in title_lower or "data" in title_lower:
                skills.extend(["Python", "SQL", "Pandas"])
            if "developer" in title_lower or "engineer" in title_lower:
                skills.extend(["Git", "AWS", "Docker"])
            if not skills:
                skills.extend(["Excel", "Communication"])

            jobs_data.append({
                "Job Title": title_text,
                "Company": company,
                "Location": location,
                "Date Posted": date_posted,
                "Skills": ", ".join(skills)
            })
            
        logging.info(f"Successfully extracted {len(jobs_data)} job postings.")
        return pd.DataFrame(jobs_data)

    def plot_top_skills_by_city(self, df: pd.DataFrame, top_n_cities: int = 3) -> None:
        """Generates a presentation-ready visualization of skill demand by region."""
        if df.empty:
            logging.warning("No data to visualize.")
            return

        # Clean and expand skills for granular analysis
        df_exploded = df.assign(Skills=df['Skills'].str.split(', ')).explode('Skills')
        
        top_cities = df_exploded['Location'].value_counts().nlargest(top_n_cities).index
        plot_data = df_exploded[df_exploded['Location'].isin(top_cities)]
        
        sns.set_theme(style="whitegrid", context="talk")
        plt.figure(figsize=(12, 6))
        
        ax = sns.countplot(
            data=plot_data, 
            x='Location', 
            hue='Skills', 
            palette='viridis',
            order=top_cities
        )
        
        plt.title("Most In-Demand Skills Across Top Job Markets", pad=20, weight='bold')
        plt.xlabel("City / Region", weight='bold')
        plt.ylabel("Frequency in Job Postings", weight='bold')
        plt.legend(title="Required Skills", bbox_to_anchor=(1.05, 1), loc='upper left')
        sns.despine()
        
        plt.tight_layout()
        plt.show()

# Execution Pipeline
if __name__ == "__main__":
    scraper = JobMarketAnalyzer(
        base_url="https://realpython.github.io",
        target_url="https://realpython.github.io/fake-jobs/"
    )
    job_df = scraper.extract_jobs()
    scraper.plot_top_skills_by_city(job_df)