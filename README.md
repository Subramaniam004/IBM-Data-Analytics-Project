# Website Phishing Risk Analytics

An interactive data analytics dashboard that turns website signals into **risk insights and recommended security actions**.

**Flow:** Fact → Insight → Decision → Action

## Business Problem

Security teams cannot manually inspect every website. Which website signals best separate phishing sites from legitimate ones, and what simple screening rules should a team apply first?

## Dataset

- **Name:** Website Phishing (UCI Machine Learning Repository)
- **Link:** https://archive.ics.uci.edu/dataset/379/website+phishing
- **Size:** 1,353 websites, 9 features (702 phishing, 103 suspicious, 548 legitimate)
- **License:** CC BY 4.0
- **Target:** `Result` (-1 = phishing, 0 = suspicious, 1 = legitimate)

## What the Dashboard Shows

| Page | Purpose |
|---|---|
| Executive Overview | KPIs (phishing rate, suspicious share, legitimate share), class distribution, top risk signals, key insights |
| Risk Drivers | Phishing rate for every signal value, class mix per signal, signal ranking |
| Risk & Actions | Phishing rate by number of phishy signals, risk/opportunity/action table, screening simulator, downloadable insights |
| Data Overview | Data quality checks, preview, class counts, feature dictionary |

All numbers are computed live from the data. This project uses descriptive analytics (no prediction model).

## Project Files

| File | Description |
|---|---|
| `app.py` | Complete dashboard (data loading, analytics and UI in one file) |
| `requirements.txt` | Python libraries |
| `README.md` | This file |
| Project report | Report with method, screenshots, insights and actions |

## How to Run

1. Install Python 3.10 or newer.
2. Install the libraries:
   ```bash
   pip install -r requirements.txt
   ```
3. Start the app:
   ```bash
   pip install streamlit pandas numpy plotly scipy ucimlrepo
   streamlit run app.py
   ```
4. The app downloads the dataset automatically from UCI. If you have no internet access, download the dataset from the link above, save it as `website_phishing.csv` next to `app.py`, or upload it in the sidebar (CSV or ARFF).

Tested with Python 3.12, Streamlit 1.65, pandas 3.0, Plotly 7.1.

## Key Insights

- 51.9% of the 1,353 websites are phishing and 7.6% are suspicious; only 40.5% are clearly legitimate.
- Server form handler is one of the strongest risk signals: 84% of sites with value 1 are phishing, versus 1% of sites with value 0.
- Pop-up window is one of the strongest risk signals: 92% of sites with value 1 are phishing, versus 23% of sites with value -1.
- SSL certificate state is one of the strongest risk signals: 74% of sites with value 1 are phishing, versus 20% of sites with value -1.
- A simple rule works well: websites with 6 or more risky signals (198 sites) are 92% phishing.
- Suspicious sites (7.6% of the data) carry mixed signals and should go to a manual review queue.

## Recommended Actions

- **Risk:** 198 sites have 6+ risky signals and 92% are phishing → Auto-flag or block websites with 6 or more risky signals
- **Risk:** Server form handler = 1: 84% phishing, covers 57% of sites → Add "Server form handler" to the first-line screening checklist
- **Risk:** Pop-up window = 1: 92% phishing, covers 13% of sites → Add "Pop-up window" to the first-line screening checklist
- **Risk:** SSL certificate state = 1: 74% phishing, covers 56% of sites → Add "SSL certificate state" to the first-line screening checklist
- **Opportunity:** 117 sites have no risky signal (phishing rate 1%) → Fast-track these sites to reduce analyst workload
- **Risk:** Suspicious sites are 7.6% of the data with mixed signals → Send to a manual review / sandbox queue

## Limitations

- Descriptive analysis only; it shows association, not causation.
- The dataset is small (1,353 sites) and from a fixed period, so patterns may change as attackers change tactics.
- Signal values are coded categories rather than raw website content.

## Credits

Dataset: Abdelhamid, Ayesh and Thabtah, via the UCI Machine Learning Repository (CC BY 4.0). See the dataset page for the introductory paper.
