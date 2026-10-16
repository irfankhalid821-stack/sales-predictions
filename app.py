
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------------------------
# 1. Page setup and visual design
# ---------------------------------
st.set_page_config(
    page_title="SmartSales AI",
    page_icon="📈",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(180deg, #f2f8ff 0%, #ffffff 65%);
}
h1, h2, h3 {
    color: #12335b;
}
div.stButton > button {
    background-color: #1769e0;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.7rem 1rem;
    font-weight: bold;
}
div.stButton > button:hover {
    background-color: #0d4fb5;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# 2. Create fictional training data
# ---------------------------------
@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    number_of_examples = 600

    advertising = rng.uniform(
        50, 2000, number_of_examples
    )
    days = rng.choice(
        [7, 14, 30, 60, 90], number_of_examples
    )

    # Fictional rule used to create practice data:
    # revenue = 180 * days + 3.2 * advertising + noise
    noise = rng.normal(0, 250, number_of_examples)

    revenue = (
        180 * days
        + 3.2 * advertising
        + noise
    )
    revenue = np.maximum(revenue, 0)

    data = pd.DataFrame({
        "Advertising_Budget": advertising,
        "Days": days,
        "Sales_Revenue": revenue
    })

    X = data[["Advertising_Budget", "Days"]]
    y = data["Sales_Revenue"]

    # Keep some examples separate for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    test_predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, test_predictions)
    r2 = r2_score(y_test, test_predictions)

    return model, mae, r2


model, mae, r2 = train_model()

# ---------------------------------
# 3. Website header
# ---------------------------------
st.title("📈 SmartSales AI")
st.subheader("Predict your sales, simply.")
st.write(
    "A friendly sales forecasting demo for small business owners. "
    "No programming knowledge needed."
)

st.info(
    "👋 Welcome! Enter your advertising budget and choose how many "
    "days you want to forecast."
)

# ---------------------------------
# 4. User-friendly input area
# ---------------------------------
left, right = st.columns([1, 1.3], gap="large")

with left:
    st.markdown("### 📝 Your business details")

    with st.form("forecast_form"):
        budget = st.number_input(
            "Advertising budget ($)",
            min_value=50,
            max_value=2000,
            value=550,
            step=50,
            help="Your total advertising budget for the entire forecast period."
        )

        days = st.selectbox(
            "Forecast period",
            options=[7, 14, 30, 60, 90],
            index=0,
            format_func=lambda d: {
                7: "7 days — one week",
                14: "14 days — two weeks",
                30: "30 days — one month",
                60: "60 days — two months",
                90: "90 days — three months"
            }[d]
        )

        submitted = st.form_submit_button(
            "✨ Predict my sales",
            use_container_width=True
        )

with right:
    st.markdown("### 💡 How it works")
    st.markdown("""
    **1. Enter your details**  
    Tell the app your advertising budget and forecast period.

    **2. Let AI learn**  
    The model learns patterns from example data.

    **3. Get a prediction**  
    See an estimated sales revenue and a simple chart.
    """)

# ---------------------------------
# 5. Make a prediction
# ---------------------------------
if submitted:
    new_business = pd.DataFrame({
        "Advertising_Budget": [budget],
        "Days": [days]
    })

    predicted_revenue = float(
        model.predict(new_business)[0]
    )
    predicted_revenue = max(0, predicted_revenue)

    st.divider()
    st.markdown("## 🎯 Your sales forecast")

    result_col, info_col = st.columns([1, 1])

    with result_col:
        st.metric(
            "Estimated total sales revenue",
            f"${predicted_revenue:,.0f}"
        )

    with info_col:
        st.metric("Forecast length", f"{days} days")
        st.caption(
            f"Advertising budget: ${budget:,.0f}"
        )

    st.success(
        f"Your demo model estimates about "
        f"${predicted_revenue:,.0f} in sales revenue "
        f"over {days} days."
    )

    # Chart: predictions for different budgets
    budget_range = np.linspace(50, 2000, 40)

    chart_data = pd.DataFrame({
        "Advertising_Budget": budget_range,
        "Days": [days] * len(budget_range)
    })

    chart_data["Predicted_Revenue"] = model.predict(
        chart_data[["Advertising_Budget", "Days"]]
    ).clip(min=0)

    chart_data = chart_data.set_index("Advertising_Budget")

    st.markdown("### 📊 How advertising budget changes the estimate")
    st.line_chart(chart_data["Predicted_Revenue"])

    st.caption(
        "The line shows model predictions at different advertising "
        "budgets for the selected forecast period."
    )

    # ---------------------------------
    # 6. Explain model performance
    # ---------------------------------
    with st.expander("🔍 How accurate is this model?"):
        st.write(
            f"**Test-set Mean Absolute Error (MAE):** "
            f"${mae:,.0f}"
        )
        st.write(f"**Test-set R² score:** {r2:.2f}")
        st.write(
            "MAE shows the average size of prediction errors on "
            "held-out practice examples. R² measures how well the "
            "model explains variation in those examples."
        )

# ---------------------------------
# 7. Important limitations
# ---------------------------------
st.divider()
with st.expander("ℹ️ Important information before using this"):
    st.warning(
        "This is a learning demo. The training data is fictional, "
        "not collected from real shops or customers."
    )
    st.write("""
    - The estimated revenue is not a guarantee of future sales.
    - The model uses only advertising budget and forecast length.
    - It does not account for product prices, seasonality, competition,
      customer demand, or other real business factors.
    - Its evaluation scores describe performance on synthetic test data,
      not real-world forecasting accuracy.
    - Predictions should not be used for financial decisions.
    """)

st.caption(
    "SmartSales AI • Beginner machine learning project • Demo data only"
)
