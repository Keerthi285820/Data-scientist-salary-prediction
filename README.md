# 🧠 Data Scientist Salary & Cost‑of‑Living Advisor (with INR & LPA)

An interactive **Streamlit web app** that predicts data science salaries across countries and adjusts them for **cost of living**.  
The app also converts salaries to **INR** and **LPA (lakhs per annum)** and computes a **“real salary score”** to estimate purchasing power.

## 🌟 Overview

Most salary dashboards show only **nominal pay** (e.g., $120k in the US vs $60k in India) and ignore how expensive it is to live in each country.

This project combines:

- a **salary prediction model** trained on real data science job records, and  
- a **country‑level cost‑of‑living index**

to answer a more useful question:

> **“For my data science profile, which countries actually give me the best *effective* salary, not just the largest number?”**

The result is a small end‑to‑end ML product: **data preparation → model → Streamlit web app**.

## 🎯 App Capabilities

### 1. Predict My Salary

Given your profile, the app predicts your expected annual salary and shows:

- **Inputs**
  - Country (company location)
  - Experience level (`EN`, `MI`, `SE`, `EX`)
  - Job title (e.g., Data Scientist, ML Engineer…)
  - Employment type (`FT`, `PT`, `CT`, `FL`)
  - Remote ratio (%)
  - Company size (`S`, `M`, `L`)

- **Outputs**
  - Predicted salary in **USD / year**
  - Predicted salary in **INR / year**
  - Predicted **LPA (lakhs per annum)**
  - **Real salary score** = `salary_in_usd / cost_of_living_index`  
    (higher = more purchasing power)

### 2. Compare Countries for the Same Profile

You can fix a single profile (same job, experience etc.) and let the app:

- Predict salary for **every country** in the dataset
- Rank countries by:
  - **Predicted salary** (USD / INR / LPA)
  - **Cost‑of‑living‑adjusted real salary score**

This shows how rankings change once you factor in living costs.

## 🧠 Model & Approach

- **Model:** `RandomForestRegressor`  
- **Features:**
  - `experience_level`
  - `employment_type`
  - `job_title`
  - `company_location`
  - `remote_ratio`
  - `company_size`
  - `cost_of_living_index`
- **Target:**
  - `salary_in_usd`

- **Pipeline:**
  - `ColumnTransformer`  
    - Numeric: `remote_ratio`, `cost_of_living_index`
    - Categorical: One‑hot encoding for the rest
  - `RandomForestRegressor` (300 trees, `random_state=42`)

- **Evaluation:**
  - Metric: **Mean Absolute Error (MAE)** on a hold‑out test set  
  - Validation MAE in this project: **≈ \$XX,XXX USD**  
    > _(Replace with your actual number from the app output.)_

The entire pipeline is trained **inside the Streamlit app** (cached with `@st.cache_resource`), so there is no separate model deployment.

## 📦 Tech Stack

- **Python**
- **Streamlit** – interactive web app
- **pandas** – data manipulation
- **scikit‑learn** – model, pipeline & preprocessing
- **pycountry** – mapping 2‑letter country codes → country names

## 📂 Project Structure

```text
.
├── app.py                         # Streamlit app (model + UI)
├── ds_salaries.csv                # Data science job salary dataset  (from Kaggle)
├── Cost_of_Living_Index_2022.csv  # Cost of living index dataset     (from Kaggle)
└── README.md
