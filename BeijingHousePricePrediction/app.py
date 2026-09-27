from pathlib import Path

import streamlit as st
import pandas as pd
import joblib

BASE_DIR = Path(__file__).resolve().parent

# ======================================================
# Load Model
# ======================================================

model = joblib.load(BASE_DIR / "xgboost_house_price.pkl")
feature_columns = joblib.load(BASE_DIR / "feature_columns.pkl")

TRADE_YEAR = 2017

# ======================================================
# Page Configuration
# ======================================================

st.set_page_config(
    page_title="Beijing Property Price Prediction | XGBoost + Streamlit",
    page_icon="🏠",
    layout="wide"
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(135deg, #f7f9fc 0%, #eef4f8 100%);
    }
    [data-testid="stSidebar"] {
        background: #102a43;
    }
    [data-testid="stSidebar"] * {
        color: #f7fbff;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.82);
        border: 1px solid #d9e2ec;
        border-radius: 8px;
        padding: 1rem;
    }
    .hero {
        background: #102a43;
        border-radius: 8px;
        color: #f7fbff;
        padding: 1.5rem 1.75rem;
        margin-bottom: 1.25rem;
    }
    .hero h1 {
        color: #f7fbff;
        margin: 0 0 0.35rem;
    }
    .hero p {
        color: #d9e2ec;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <section class="hero">
        <h1>Beijing Property Price Prediction | XGBoost + Streamlit</h1>
        <p>Explore an XGBoost estimate from a few practical property details.</p>
    </section>
    """,
    unsafe_allow_html=True,
)

st.caption("Model: XGBoost Regressor  •  Currency: Chinese yuan (RMB)  •  Reference year: 2017")

# ======================================================
# Sidebar
# ======================================================

st.sidebar.header("🏡 Property Details")
st.sidebar.caption("Adjust the inputs, then run the estimate.")
st.sidebar.markdown("**Size and layout**")

square = st.sidebar.number_input(
    "Square Area (m²)",
    min_value=20.0,
    max_value=500.0,
    value=90.0
)

living = st.sidebar.slider(
    "Living Rooms",
    1,
    8,
    2
)

bath = st.sidebar.slider(
    "Bathrooms",
    1,
    5,
    1
)

st.sidebar.markdown("**Building details**")

construction = st.sidebar.number_input(
    "Construction Year",
    min_value=1950,
    max_value=TRADE_YEAR,
    value=2010
)

floor = st.sidebar.number_input(
    "Floor Number",
    min_value=1,
    max_value=60,
    value=10
)

st.sidebar.markdown("**Location and amenities**")

community = st.sidebar.number_input(
    "Community Average Price",
    min_value=10000.0,
    max_value=200000.0,
    value=65000.0
)

district_names = {
    "Dongcheng": 1,
    "Xicheng": 2,
    "Chaoyang": 3,
    "Fengtai": 4,
    "Shijingshan": 5,
    "Haidian": 6,
    "Mentougou": 7,
    "Fangshan": 8,
    "Tongzhou": 9,
    "Shunyi": 10,
    "Changping": 11,
    "Daxing": 12,
    "Yizhuang": 13
}

district_name = st.sidebar.selectbox(
    "District",
    list(district_names.keys())
)

district = district_names[district_name]

subway = st.sidebar.selectbox(
    "Near Subway",
    ["Yes", "No"]
)

elevator = st.sidebar.selectbox(
    "Elevator",
    ["Yes", "No"]
)

predict = st.sidebar.button("🔮 Predict House Price")

# ======================================================
# Prediction
# ======================================================

if predict:

    subway_value = 1 if subway == "Yes" else 0
    elevator_value = 1 if elevator == "Yes" else 0

    house_age = TRADE_YEAR - construction

    input_data = {
        "Lng": 116.42,
        "Lat": 39.95,
        "followers": 17,
        "square": square,
        "livingRoom": living,
        "drawingRoom": 1,
        "kitchen": 1,
        "bathRoom": bath,
        "buildingType": 4,
        "constructionTime": construction,
        "renovationCondition": 1,
        "buildingStructure": 6,
        "ladderRatio": 63.17,
        "elevator": elevator_value,
        "fiveYearsProperty": 1,
        "subway": subway_value,
        "district": district,
        "communityAverage": community,
        "tradeYear": TRADE_YEAR,
        "tradeMonth": 6,
        "tradeQuarter": 2,
        "houseAge": house_age,
        "floorNumber": floor
    }

    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]

    with st.spinner("Predicting House Price..."):

        prediction = model.predict(input_df)[0]

    # Convert to RMB
    price_rmb = prediction * 10000
    price_million = price_rmb / 1_000_000

    # Property Category
    if price_million < 3:
        category = "🟢 Budget"
    elif price_million < 8:
        category = "🟡 Mid-range"
    else:
        category = "🔴 Luxury"

    st.success("Prediction Completed Successfully!")

    # ==================================================
    # Metrics
    # ==================================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Estimated Price",
            f"¥ {price_rmb:,.0f}"
        )

    with col2:
        st.metric(
            "Price (Million RMB)",
            f"{price_million:.2f} M"
        )

    with col3:
        st.metric(
            "Property Category",
            category
        )

    st.divider()

    # ==================================================
    # Property Summary
    # ==================================================

    st.subheader("🏡 Property Summary")

    summary = pd.DataFrame({

        "Feature": [
            "Square Area",
            "Living Rooms",
            "Bathrooms",
            "Construction Year",
            "House Age",
            "Floor Number",
            "District",
            "Near Subway",
            "Elevator",
            "Community Average Price"
        ],

        "Value": [
            f"{square:.2f} m²",
            living,
            bath,
            construction,
            house_age,
            floor,
            district_name,
            subway,
            elevator,
            f"¥ {community:,.0f}"
        ]

    })

    st.dataframe(
        summary,
        hide_index=True,
        use_container_width=True
    )

    st.divider()

    # ==================================================
    # Model Performance
    # ==================================================

    st.subheader("🤖 Model Performance")

    performance = pd.DataFrame({

        "Metric": [
            "Model",
            "R² Score",
            "RMSE",
            "MAE"
        ],

        "Value": [
            "XGBoost Regressor",
            "94.69%",
            "52.98",
            "28.34"
        ]

    })

    st.dataframe(
        performance,
        hide_index=True,
        use_container_width=True
    )
else:
    overview_col1, overview_col2, overview_col3 = st.columns(3)

    with overview_col1:
        st.metric("Area range", "20–500 m²")
    with overview_col2:
        st.metric("Districts", "13 covered")
    with overview_col3:
        st.metric("Prediction model", "XGBoost")

    st.info("Choose property details in the sidebar and select **Predict House Price** to see the estimate and property summary.")

# ======================================================
# Footer
# ======================================================

st.markdown("---")

st.markdown("""
### Beijing Property Price Prediction | XGBoost + Streamlit

**Project Type:** Machine Learning Project

**Machine Learning Model:** XGBoost Regressor

**Dataset:** Beijing Housing Dataset

**Technologies Used**

- Python
- Streamlit
- XGBoost
- Pandas
- Scikit-learn

Developed as part of a Machine Learning and Cloud Computing Project.
""")