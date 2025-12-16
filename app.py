import streamlit as st
import pandas as pd
import pycountry

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# -----------------------------
# SETTINGS
# -----------------------------
# Approximate USD → INR rate
USD_TO_INR = 83.0  # change if you want a more recent rate


# ---------------------------------------------------------
# Data loading + model training (runs once, then cached)
# ---------------------------------------------------------
@st.cache_resource
def load_data_and_train():
    # ---- 1. Load raw CSVs ----
    # make sure these 2 files are in the SAME folder as app.py
    salaries = pd.read_csv("ds_salaries.csv")
    col = pd.read_csv("Cost_of_Living_Index_2022.csv")  # adjust name if your file differs

    # ---- 2. Map country codes to country names ----
    def code_to_country(code):
        try:
            c = pycountry.countries.get(alpha_2=code)
            return c.name if c else None
        except Exception:
            return None

    salaries["country"] = salaries["company_location"].apply(code_to_country)

    use_cols = [
        "country",
        "company_location",
        "experience_level",
        "employment_type",
        "job_title",
        "remote_ratio",
        "company_size",
        "salary_in_usd",
    ]
    salaries = salaries[use_cols].dropna(subset=["country"])

    # ---- 3. Prepare cost-of-living data ----
    orig_cols = list(col.columns)

    country_col = None
    col_index_col = None
    for c in orig_cols:
        lc = c.lower()
        if "country" in lc:
            country_col = c
        if "cost" in lc and "living" in lc:
            col_index_col = c

    if country_col is None or col_index_col is None:
        raise ValueError(
            f"Could not find country / cost-of-living columns. Columns are: {orig_cols}"
        )

    col_simple = col[[country_col, col_index_col]].copy()
    col_simple.columns = ["country", "cost_of_living_index"]
    col_simple["country"] = col_simple["country"].astype(str).str.strip()
    col_simple = col_simple.dropna(subset=["country", "cost_of_living_index"])

    # ---- 4. Merge salaries + COL ----
    df = pd.merge(salaries, col_simple, on="country", how="inner")

    # Save lookup for UI
    country_lookup = df[
        ["country", "company_location", "cost_of_living_index"]
    ].drop_duplicates()

    # Take most common job titles for dropdown
    job_titles = df["job_title"].value_counts().head(40).index.tolist()

    # ---- 5. Train model ----
    features = [
        "experience_level",
        "employment_type",
        "job_title",
        "company_location",
        "remote_ratio",
        "company_size",
        "cost_of_living_index",
    ]
    target = "salary_in_usd"

    X = df[features]
    y = df[target]

    numeric_features = ["remote_ratio", "cost_of_living_index"]
    categorical_features = [
        "experience_level",
        "employment_type",
        "job_title",
        "company_location",
        "company_size",
    ]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300, random_state=42, n_jobs=-1
    )

    pipe = Pipeline(steps=[("preprocess", preprocess), ("model", model)])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)

    return pipe, mae, country_lookup, job_titles


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def get_cost_of_living(country_df, country_name):
    row = country_df[country_df["country"] == country_name]
    if row.empty:
        return None
    return float(row["cost_of_living_index"].iloc[0])


def get_country_code(country_df, country_name):
    row = country_df[country_df["country"] == country_name]
    if row.empty:
        return None
    return row["company_location"].iloc[0]


# ---------------------------------------------------------
# Streamlit App
# ---------------------------------------------------------
st.set_page_config(page_title="DS Salary & COL Advisor", layout="wide")
st.title("🧠 Data Scientist Salary & Cost-of-Living Advisor")

st.write(
    "This app uses the **Data Science Job Salaries** dataset plus a "
    "**2022 Cost-of-Living Index** to predict data science salaries and "
    "estimate real purchasing power across countries."
)

st.info("Loading data and training model... (runs once, then cached)")
pipe, mae, country_df, job_titles = load_data_and_train()
st.success(f"Model ready. Validation MAE ≈ ${mae:,.0f} USD.")

tab1, tab2 = st.tabs(["🔮 Predict my salary", "🌍 Compare countries"])


# ---------------------------------------------------------
# TAB 1 – Predict salary for one profile
# ---------------------------------------------------------
with tab1:
    st.subheader("Predict expected salary for your profile")

    col_left, col_right = st.columns(2)

    with col_left:
        country = st.selectbox(
            "Country (company location)",
            sorted(country_df["country"].unique()),
        )

        experience_level = st.selectbox(
            "Experience level",
            ["EN", "MI", "SE", "EX"],
            help="EN=Entry, MI=Mid, SE=Senior, EX=Executive",
        )

        employment_type = st.selectbox(
            "Employment type",
            ["FT", "PT", "CT", "FL"],
            help="FT=Full-time, PT=Part-time, CT=Contract, FL=Freelance",
        )

        company_size = st.selectbox(
            "Company size",
            ["S", "M", "L"],
            help="S=Small, M=Medium, L=Large",
        )

    with col_right:
        job_title = st.selectbox("Job title", job_titles)
        remote_ratio = st.slider(
            "Remote ratio (%)", min_value=0, max_value=100, step=25, value=0
        )

    if st.button("Predict my salary"):
        col_index = get_cost_of_living(country_df, country)
        company_code = get_country_code(country_df, country)

        if col_index is None or company_code is None:
            st.error("No cost-of-living data for this country.")
        else:
            input_df = pd.DataFrame(
                [
                    {
                        "experience_level": experience_level,
                        "employment_type": employment_type,
                        "job_title": job_title,
                        "company_location": company_code,
                        "remote_ratio": remote_ratio,
                        "company_size": company_size,
                        "cost_of_living_index": col_index,
                    }
                ]
            )

            pred_salary = pipe.predict(input_df)[0]
            real_index = pred_salary / col_index

            # --- INR + LPA conversions ---
            inr_salary = pred_salary * USD_TO_INR       # total INR per year
            lpa = inr_salary / 100000.0                 # lakhs per annum

            st.success(
                f"Predicted annual salary: **${pred_salary:,.0f} USD**  "
                f"(≈ ₹{inr_salary:,.0f} / year, ~{lpa:.2f} LPA)"
            )
            st.write(
                f"Cost-of-living index for **{country}**: `{col_index:.1f}` "
                "(higher = more expensive)."
            )

            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Predicted salary (USD)", f"${pred_salary:,.0f}")
            with c2:
                st.metric("Predicted salary (INR)", f"₹{inr_salary:,.0f}")
            with c3:
                st.metric("Salary (LPA)", f"{lpa:.2f} LPA")

            st.metric(
                "Real salary score",
                f"{real_index:.1f}",
                help="Predicted salary divided by cost-of-living index "
                     "(higher = more purchasing power).",
            )


# ---------------------------------------------------------
# TAB 2 – Compare countries for same profile
# ---------------------------------------------------------
with tab2:
    st.subheader("Find the best countries for your profile")

    c1, c2, c3 = st.columns(3)
    with c1:
        experience_level2 = st.selectbox(
            "Experience level", ["EN", "MI", "SE", "EX"], key="exp2"
        )
        employment_type2 = st.selectbox(
            "Employment type", ["FT", "PT", "CT", "FL"], key="emp2"
        )
    with c2:
        job_title2 = st.selectbox("Job title", job_titles, key="job2")
        company_size2 = st.selectbox("Company size", ["S", "M", "L"], key="size2")
    with c3:
        remote_ratio2 = st.slider(
            "Remote ratio (%)", 0, 100, 0, 25, key="rem2"
        )
        top_n = st.slider("Show top N countries", 5, 20, 10)

    if st.button("Compare countries"):
        rows = []
        for _, row in country_df.iterrows():
            c_name = row["country"]
            col_index = row["cost_of_living_index"]
            c_code = row["company_location"]

            input_df = pd.DataFrame(
                [
                    {
                        "experience_level": experience_level2,
                        "employment_type": employment_type2,
                        "job_title": job_title2,
                        "company_location": c_code,
                        "remote_ratio": remote_ratio2,
                        "company_size": company_size2,
                        "cost_of_living_index": col_index,
                    }
                ]
            )

            salary_pred = pipe.predict(input_df)[0]
            real_idx = salary_pred / col_index

            inr_salary = salary_pred * USD_TO_INR
            lpa = inr_salary / 100000.0

            rows.append(
                {
                    "country": c_name,
                    "pred_salary_usd": salary_pred,
                    "pred_salary_inr": inr_salary,
                    "salary_lpa": lpa,
                    "cost_of_living_index": col_index,
                    "real_salary_index": real_idx,
                }
            )

        compare_df = pd.DataFrame(rows)

        top_nominal = compare_df.sort_values(
            "pred_salary_usd", ascending=False
        ).head(top_n)
        top_real = compare_df.sort_values(
            "real_salary_index", ascending=False
        ).head(top_n)

        st.markdown("### Top countries by predicted salary")
        st.dataframe(
            top_nominal.set_index("country").style.format(
                {
                    "pred_salary_usd": "{:,.0f}",
                    "pred_salary_inr": "₹{:,.0f}",
                    "salary_lpa": "{:.2f} LPA",
                    "cost_of_living_index": "{:.1f}",
                }
            )
        )
        st.bar_chart(top_nominal.set_index("country")["pred_salary_usd"])

        st.markdown("### Top countries by predicted **real** salary (salary / cost of living)")
        st.dataframe(
            top_real.set_index("country").style.format(
                {
                    "pred_salary_usd": "{:,.0f}",
                    "pred_salary_inr": "₹{:,.0f}",
                    "salary_lpa": "{:.2f} LPA",
                    "cost_of_living_index": "{:.1f}",
                    "real_salary_index": "{:.1f}",
                }
            )
        )
        st.bar_chart(top_real.set_index("country")["real_salary_index"])