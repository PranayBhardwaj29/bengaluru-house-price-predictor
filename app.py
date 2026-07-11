import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(
    page_title="Bengaluru Home Price Estimator",
    page_icon="🏡",
    layout="centered",
)

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #12312B 0%, #16403A 55%, #12312B 100%);
    }

    /* Hero */
    .hero-eyebrow {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.18em;
        text-transform: uppercase;
        color: #C9A227;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }
    .hero-title {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 2.6rem;
        color: #F7F4EE;
        line-height: 1.05;
        margin-bottom: 0.5rem;
    }
    .hero-sub {
        font-family: 'Inter', sans-serif;
        color: #B9CBC3;
        font-size: 1.02rem;
        margin-bottom: 1.8rem;
    }

    /* Card */
    .card {
        background: #F7F4EE;
        border-radius: 18px;
        padding: 1.9rem 1.9rem 1.4rem 1.9rem;
        box-shadow: 0 12px 30px rgba(0,0,0,0.25);
        margin-bottom: 1.6rem;
    }
    .card-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        letter-spacing: 0.14em;
        text-transform: uppercase;
        color: #7FA294;
        font-weight: 700;
        margin-bottom: 0.6rem;
    }

    /* Result panel */
    .result-card {
        background: linear-gradient(135deg, #1E2422 0%, #16403A 100%);
        border-radius: 18px;
        padding: 1.8rem 1.9rem;
        margin-bottom: 1.6rem;
        border: 1px solid rgba(201,162,39,0.35);
    }
    .result-label {
        font-family: 'Inter', sans-serif;
        font-size: 0.78rem;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        color: #B9CBC3;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .result-value {
        font-family: 'Fraunces', serif;
        font-weight: 700;
        font-size: 3rem;
        color: #E9C46A;
        line-height: 1;
        margin-bottom: 0.3rem;
    }
    .result-caption {
        font-family: 'Inter', sans-serif;
        color: #B9CBC3;
        font-size: 0.9rem;
    }

    /* Gauge */
    .gauge-wrap {
        margin-top: 1.1rem;
    }
    .gauge-labels {
        display: flex;
        justify-content: space-between;
        font-family: 'Inter', sans-serif;
        font-size: 0.75rem;
        color: #B9CBC3;
        margin-bottom: 0.35rem;
    }
    .gauge-track {
        position: relative;
        height: 10px;
        border-radius: 6px;
        background: linear-gradient(90deg, #2C5F53 0%, #7FA294 50%, #E9C46A 100%);
    }
    .gauge-marker {
        position: absolute;
        top: -6px;
        width: 3px;
        height: 22px;
        background: #F7F4EE;
        border-radius: 2px;
        box-shadow: 0 0 6px rgba(0,0,0,0.4);
    }

    /* Streamlit widget tweaks */
    .stButton>button {
        background: #C9A227;
        color: #1E2422;
        font-weight: 700;
        border: none;
        border-radius: 10px;
        padding: 0.6rem 1.6rem;
        font-family: 'Inter', sans-serif;
        width: 100%;
    }
    .stButton>button:hover {
        background: #E9C46A;
        color: #1E2422;
    }
    label, .stSelectbox label, .stNumberInput label, .stTextInput label {
        color: #F7F4EE !important;
        font-weight: 600 !important;
        font-size: 0.85rem !important;
    }
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Load model + transformer
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("price_model.pkl")
    ct = joblib.load("column_transformer.pkl")
    return model, ct

model, ct = load_artifacts()

# ----------------------------------------------------------------------------
# Hero
# ----------------------------------------------------------------------------
st.markdown('<div class="hero-eyebrow">BENGALURU · PROPERTY VALUATION</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-title">What\'s this home worth?</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Enter a few details about the property and get an instant price estimate, '
    'trained on cleaned Bengaluru housing data.</div>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# Input card
# ----------------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.markdown('<div class="card-label">Property Details</div>', unsafe_allow_html=True)

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        area_type = st.selectbox(
            "Area Type",
            ["Super built-up  Area", "Built-up  Area", "Plot  Area", "Carpet  Area"],
        )
        location = st.text_input("Location", placeholder="e.g. Whitefield")
        bhk = st.number_input("BHK", min_value=1, max_value=10, value=2, step=1)

    with col2:
        total_sqft = st.number_input("Total Sqft", min_value=100.0, max_value=10000.0, value=1000.0, step=50.0)
        bathrooms = st.number_input("Bathrooms", min_value=1.0, max_value=10.0, value=2.0, step=1.0)
        balcony = st.number_input("Balcony", min_value=0.0, max_value=5.0, value=1.0, step=1.0)

    availability = st.selectbox("Availability", ["Ready To Move", "Not Ready / Under Construction"])
    availability_val = 1 if availability == "Ready To Move" else 0

    submitted = st.form_submit_button("Estimate Price")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# Prediction + result
# ----------------------------------------------------------------------------
if submitted:
    if not location.strip():
        st.warning("Please enter a location.")
    else:
        input_df = pd.DataFrame([{
            "area_type": area_type,
            "availability": availability_val,
            "location": location.strip(),
            "total_sqft": total_sqft,
            "bathrooms": bathrooms,
            "balcony": balcony,
            "bhk": bhk,
        }])

        try:
            transformed = ct.transform(input_df)
            log_pred = model.predict(transformed)[0]
            price_pred = float(np.expm1(log_pred))

            # position on a simple visual scale for the gauge (10L to 400L, capped)
            low, high = 10, 400
            pct = min(max((price_pred - low) / (high - low), 0), 1) * 100

            known_locations = ct.named_transformers_["ohe"].categories_[1]
            unseen_location = location.strip() not in known_locations

            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.markdown('<div class="result-label">Estimated Price</div>', unsafe_allow_html=True)
            st.markdown(f'<div class="result-value">₹{price_pred:,.2f} L</div>', unsafe_allow_html=True)
            st.markdown(
                '<div class="result-caption">Based on a Linear Regression model (R² ≈ 0.844) '
                'trained on cleaned Bengaluru housing data.</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="gauge-wrap">
                    <div class="gauge-labels"><span>₹10L</span><span>₹400L+</span></div>
                    <div class="gauge-track">
                        <div class="gauge-marker" style="left: calc({pct}% - 1.5px);"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('</div>', unsafe_allow_html=True)

            if unseen_location:
                st.info(
                    "This location wasn't seen during training, so the estimate doesn't include "
                    "any location-specific adjustment — treat it as a rough baseline."
                )

        except Exception as e:
            st.error(f"Something went wrong during prediction: {e}")

st.markdown(
    '<div style="text-align:center; color:#7FA294; font-size:0.8rem; margin-top:1rem;">'
    'Built with Streamlit · Model trained on cleaned Bengaluru housing data</div>',
    unsafe_allow_html=True,
)