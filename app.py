# ============================================================
# GEOVALUE AI — STREAMLIT APPLICATION
# CatBoost-based House Price Prediction & Real Estate Analytics
# ============================================================

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from catboost import CatBoostRegressor


# ------------------------------------------------------------
# 1. PAGE CONFIGURATION AND PATHS
# ------------------------------------------------------------

st.set_page_config(
    page_title="GeoValue AI",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"
MODEL_DIR = ARTIFACT_DIR / "models"

CONFIG_PATH = ARTIFACT_DIR / "model_config.json"
MODEL_PATH = MODEL_DIR / "geovalue_model.cbm"
LOCALITIES_PATH = ARTIFACT_DIR / "localities.json"
TRAINING_REFERENCE_PATH = ARTIFACT_DIR / "training_reference.csv"
SHAP_REFERENCE_PATH = ARTIFACT_DIR / "shap_reference.csv"


# ------------------------------------------------------------
# 2. ARTIFACT VALIDATION AND LOADING
# ------------------------------------------------------------

def require_file(path: Path, description: str) -> None:
    """Stop with a clear message if a required artifact is missing."""
    if not path.is_file():
        st.error(
            f"Required {description} was not found:\n\n"
            f"`{path.relative_to(BASE_DIR)}`"
        )
        st.info(
            "Check that you copied the artifact into the correct "
            "location inside the GeoValue-AI project folder."
        )
        st.stop()


@st.cache_data
def load_config():
    require_file(CONFIG_PATH, "model configuration file")

    try:
        with CONFIG_PATH.open("r", encoding="utf-8") as file:
            config_data = json.load(file)
    except (json.JSONDecodeError, OSError) as exc:
        st.error(f"Could not read model_config.json: {exc}")
        st.stop()

    required_keys = [
        "model_name",
        "feature_columns",
        "categorical_features",
    ]
    missing = [key for key in required_keys if key not in config_data]
    if missing:
        st.error(
            "model_config.json is missing required keys: "
            + ", ".join(missing)
        )
        st.stop()

    if config_data["model_name"] != "CatBoost":
        st.error(
            "This application version is configured for CatBoost. "
            f"The configuration currently says: {config_data['model_name']!r}."
        )
        st.stop()

    if not isinstance(config_data["feature_columns"], list):
        st.error("'feature_columns' in model_config.json must be a list.")
        st.stop()

    if not isinstance(config_data["categorical_features"], list):
        st.error("'categorical_features' in model_config.json must be a list.")
        st.stop()

    return config_data


@st.cache_resource
def load_model():
    require_file(MODEL_PATH, "CatBoost model")

    model = CatBoostRegressor()
    try:
        model.load_model(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load the CatBoost model: {exc}")
        st.stop()

    return model


@st.cache_data
def load_reference_data():
    require_file(TRAINING_REFERENCE_PATH, "training reference CSV")
    require_file(LOCALITIES_PATH, "localities JSON")

    try:
        reference = pd.read_csv(TRAINING_REFERENCE_PATH)
        with LOCALITIES_PATH.open("r", encoding="utf-8") as file:
            localities_data = json.load(file)
    except Exception as exc:
        st.error(f"Could not load the reference data: {exc}")
        st.stop()

    required_reference_columns = [
        "location",
        "area_type",
        "total_sqft",
        "bhk",
        "bath",
        "balcony",
        "price",
    ]
    missing = [
        col for col in required_reference_columns
        if col not in reference.columns
    ]
    if missing:
        st.error(
            "training_reference.csv is missing columns: "
            + ", ".join(missing)
        )
        st.stop()

    # Support either a JSON list or a dictionary containing a list.
    if isinstance(localities_data, dict):
        if "localities" in localities_data:
            localities_data = localities_data["localities"]
        else:
            localities_data = list(localities_data.keys())

    if not isinstance(localities_data, list):
        st.error(
            "localities.json must contain a list of locality names "
            "or a dictionary with a 'localities' list."
        )
        st.stop()

    localities = sorted({
        str(item).strip()
        for item in localities_data
        if item is not None and str(item).strip()
    })

    if not localities:
        localities = sorted(
            reference["location"].dropna().astype(str).str.strip().unique()
        )

    # Normalize display/analytics columns without changing model artifacts.
    reference["location"] = reference["location"].astype(str).str.strip()
    reference["area_type"] = reference["area_type"].fillna("Unknown").astype(str)

    for col in ["total_sqft", "bhk", "bath", "balcony", "price"]:
        reference[col] = pd.to_numeric(reference[col], errors="coerce")

    reference = reference.dropna(
        subset=["location", "total_sqft", "bhk", "price"]
    )
    reference = reference[
        (reference["total_sqft"] > 0)
        & (reference["bhk"] > 0)
        & (reference["price"] > 0)
    ].copy()

    if reference.empty:
        st.error("training_reference.csv contains no usable records.")
        st.stop()

    return reference, localities


@st.cache_data
def load_shap_reference():
    require_file(SHAP_REFERENCE_PATH, "SHAP reference CSV")

    try:
        shap_reference = pd.read_csv(SHAP_REFERENCE_PATH)
    except Exception as exc:
        st.error(f"Could not read shap_reference.csv: {exc}")
        st.stop()

    return shap_reference


config = load_config()
model = load_model()
reference_df, known_localities = load_reference_data()

FEATURE_COLUMNS = config["feature_columns"]
CATEGORICAL_FEATURES = config["categorical_features"]

missing_cat_features = [
    col for col in CATEGORICAL_FEATURES if col not in FEATURE_COLUMNS
]
if missing_cat_features:
    st.error(
        "Categorical features are not present in feature_columns: "
        + ", ".join(missing_cat_features)
    )
    st.stop()

model_feature_names = model.feature_names_
if model_feature_names:
    if list(model_feature_names) != list(FEATURE_COLUMNS):
        st.error(
            "The feature order in model_config.json does not match "
            "the feature order saved in the CatBoost model.\n\n"
            f"Model features: {model_feature_names}\n\n"
            f"Config features: {FEATURE_COLUMNS}"
        )
        st.stop()

conformal_quantile = config.get("conformal_quantile")
try:
    conformal_quantile = (
        float(conformal_quantile)
        if conformal_quantile is not None
        else None
    )
except (TypeError, ValueError):
    conformal_quantile = None

if (
    conformal_quantile is not None
    and (
        not np.isfinite(conformal_quantile)
        or conformal_quantile < 0
    )
):
    conformal_quantile = None


# ------------------------------------------------------------
# 3. APPLICATION HEADER
# ------------------------------------------------------------

st.title("🏡 GeoValue AI")
st.caption(
    "Intelligent House Price Prediction and "
    "Real Estate Analytics System"
)
st.markdown(
    "Estimate residential property listing prices, explore "
    "locality-level listing patterns and understand the factors "
    "behind model predictions."
)
st.divider()


# ------------------------------------------------------------
# 4. SIDEBAR NAVIGATION
# ------------------------------------------------------------

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "House Price Prediction",
        "Market Analytics",
        "Similar Properties",
        "Explainable AI",
        "About GeoValue AI",
    ],
)

st.sidebar.divider()
st.sidebar.caption("GeoValue AI | Academic ML Project")


# ------------------------------------------------------------
# 5. FEATURE PREPARATION
# ------------------------------------------------------------

def prepare_input(
    location,
    area_type,
    availability_status,
    total_sqft,
    bhk,
    bath,
    balcony,
):
    """Build a one-row DataFrame in the exact model feature order."""
    if float(total_sqft) <= 0 or float(bhk) <= 0:
        raise ValueError("Area and BHK must be greater than zero.")

    bath_value = np.nan if bath is None else float(bath)
    balcony_value = np.nan if balcony is None else float(balcony)

    row = {
        "location": str(location).strip(),
        "area_type": str(area_type),
        "availability_status": str(availability_status),
        "total_sqft": float(total_sqft),
        "bhk": float(bhk),
        "bath": bath_value,
        "balcony": balcony_value,
        "area_per_bhk": float(total_sqft) / float(bhk),
        "bath_per_bhk": (
            bath_value / float(bhk)
            if pd.notna(bath_value)
            else np.nan
        ),
        "balcony_per_bhk": (
            balcony_value / float(bhk)
            if pd.notna(balcony_value)
            else np.nan
        ),
    }

    input_data = pd.DataFrame([row])

    missing_features = [
        col for col in FEATURE_COLUMNS if col not in input_data.columns
    ]
    if missing_features:
        raise ValueError(
            "The application cannot construct these model features: "
            + ", ".join(missing_features)
        )

    input_data = input_data[FEATURE_COLUMNS].copy()

    for column in CATEGORICAL_FEATURES:
        input_data[column] = (
            input_data[column]
            .fillna("__MISSING__")
            .astype(str)
        )

    return input_data


# ------------------------------------------------------------
# 6. PRICE PREDICTION
# ------------------------------------------------------------

def predict_property(input_data):
    prediction = float(model.predict(input_data)[0])

    if not np.isfinite(prediction):
        raise ValueError("The model returned an invalid prediction.")

    lower = None
    upper = None

    if conformal_quantile is not None:
        lower = max(0.0, prediction - conformal_quantile)
        upper = prediction + conformal_quantile

    return prediction, lower, upper


# ------------------------------------------------------------
# 7. DASHBOARD PAGE
# ------------------------------------------------------------

if page == "Dashboard":

    st.header("Real Estate Market Dashboard")

    col1, col2, col3, col4 = st.columns(4)

    total_properties = len(reference_df)
    total_localities = reference_df["location"].nunique()
    average_price = reference_df["price"].mean()
    median_price = reference_df["price"].median()

    col1.metric("Training Properties", f"{total_properties:,}")
    col2.metric("Training Localities", f"{total_localities:,}")
    col3.metric("Average Listing Price", f"₹{average_price:.2f} L")
    col4.metric("Median Listing Price", f"₹{median_price:.2f} L")

    st.divider()

    left, right = st.columns(2)

    with left:
        st.subheader("Price Distribution")
        fig = px.histogram(
            reference_df,
            x="price",
            nbins=40,
            title="Property Listing Price Distribution",
        )
        fig.update_layout(
            xaxis_title="Price (₹ Lakhs)",
            yaxis_title="Number of Listings",
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        st.subheader("Top Localities by Listings")
        top_locations = (
            reference_df["location"]
            .value_counts()
            .head(10)
            .rename_axis("Location")
            .reset_index(name="Listings")
        )
        fig = px.bar(
            top_locations,
            x="Listings",
            y="Location",
            orientation="h",
            title="Top 10 Localities",
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Median Listing Price by Locality")
    locality_summary = (
        reference_df.groupby("location")
        .agg(
            Median_Price=("price", "median"),
            Listing_Count=("price", "count"),
        )
        .reset_index()
    )
    locality_summary = locality_summary[
        locality_summary["Listing_Count"] >= 5
    ].sort_values("Median_Price", ascending=False)

    st.dataframe(
        locality_summary.head(20),
        use_container_width=True,
        hide_index=True,
    )

    st.caption(
        "Dashboard statistics describe the saved training reference "
        "listings, not verified property transactions or live market prices."
    )


# ------------------------------------------------------------
# 8. HOUSE PRICE PREDICTION PAGE
# ------------------------------------------------------------

elif page == "House Price Prediction":

    st.header("House Price Prediction")
    st.write(
        "Enter the residential property details to estimate its listing price."
    )

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        with col1:
            location_options = ["Other / Unseen Locality"] + known_localities
            selected_location = st.selectbox(
                "Property Location",
                location_options,
            )

            if selected_location == "Other / Unseen Locality":
                location = st.text_input(
                    "Enter locality name",
                    placeholder="e.g. Whitefield",
                ).strip()
            else:
                location = selected_location

            area_types = sorted(
                reference_df["area_type"].dropna().astype(str).unique().tolist()
            )
            area_type = st.selectbox("Area Type", area_types)

            total_sqft = st.number_input(
                "Total Area (sq. ft.)",
                min_value=100.0,
                max_value=50000.0,
                value=1200.0,
                step=50.0,
            )

            bhk = st.number_input(
                "BHK",
                min_value=1,
                max_value=15,
                value=2,
                step=1,
            )

        with col2:
            bath = st.number_input(
                "Bathrooms",
                min_value=0,
                max_value=20,
                value=2,
                step=1,
            )

            balcony = st.number_input(
                "Balconies",
                min_value=0,
                max_value=10,
                value=1,
                step=1,
            )

            availability_status = st.selectbox(
                "Availability",
                ["Ready To Move", "Other Availability"],
            )

        submitted = st.form_submit_button(
            "Predict Property Price",
            use_container_width=True,
        )

    if submitted:
        if not location:
            st.warning("Enter a locality name before requesting a prediction.")
        elif total_sqft / bhk < 100:
            st.warning(
                "The entered area per BHK is unusually low. "
                "Please verify your inputs."
            )
        else:
            try:
                input_data = prepare_input(
                    location=location,
                    area_type=area_type,
                    availability_status=availability_status,
                    total_sqft=total_sqft,
                    bhk=bhk,
                    bath=bath,
                    balcony=balcony,
                )

                prediction, lower, upper = predict_property(input_data)

                st.divider()
                st.subheader("Prediction Result")
                st.metric(
                    "Estimated Property Listing Price",
                    f"₹{prediction:.2f} Lakh",
                )

                if lower is not None:
                    st.info(
                        f"Calibrated prediction interval "
                        f"(nominal 90%): ₹{lower:.2f} Lakh "
                        f"to ₹{upper:.2f} Lakh"
                    )
                else:
                    st.warning(
                        "A finite calibrated prediction interval is "
                        "unavailable. Only the point estimate is shown."
                    )

                predicted_price_per_sqft = (
                    prediction * 100000.0 / total_sqft
                )

                col1, col2, col3 = st.columns(3)
                col1.metric(
                    "Predicted Price per Sq. Ft.",
                    f"₹{predicted_price_per_sqft:,.0f}",
                )
                col2.metric("Property Area", f"{total_sqft:,.0f} sq. ft.")
                col3.metric("Configuration", f"{bhk} BHK")

                if location not in known_localities:
                    st.warning(
                        "This locality was not represented in the training "
                        "locality list. The prediction may be less reliable."
                    )

                st.caption(
                    "This is a model estimate based on historical dataset "
                    "listings, not a certified property valuation."
                )

            except Exception as error:
                st.error(f"Prediction failed: {error}")


# ------------------------------------------------------------
# 9. MARKET ANALYTICS PAGE
# ------------------------------------------------------------

elif page == "Market Analytics":

    st.header("Market Analytics")

    selected_location = st.selectbox(
        "Select Locality",
        ["All Localities"] + known_localities,
    )

    filtered_df = reference_df.copy()

    if selected_location != "All Localities":
        filtered_df = filtered_df[
            filtered_df["location"] == selected_location
        ]

    if filtered_df.empty:
        st.warning("No reference listings are available for this selection.")
    else:
        col1, col2, col3 = st.columns(3)

        col1.metric("Listings", f"{len(filtered_df):,}")
        col2.metric(
            "Median Price",
            f"₹{filtered_df['price'].median():.2f} L",
        )
        col3.metric(
            "Average Area",
            f"{filtered_df['total_sqft'].mean():,.0f} sq. ft.",
        )

        fig = px.scatter(
            filtered_df,
            x="total_sqft",
            y="price",
            color="bhk",
            hover_data=["location"],
            title="Property Area vs Listing Price",
        )
        fig.update_layout(
            xaxis_title="Area (sq. ft.)",
            yaxis_title="Listing Price (₹ Lakhs)",
        )
        st.plotly_chart(fig, use_container_width=True)

        fig = px.box(
            filtered_df,
            x="bhk",
            y="price",
            title="Price Distribution by BHK",
        )
        fig.update_layout(
            xaxis_title="BHK",
            yaxis_title="Listing Price (₹ Lakhs)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.dataframe(
            filtered_df.head(100),
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        "Analytics are calculated from the saved training reference dataset; "
        "they are not live market statistics."
    )


# ------------------------------------------------------------
# 10. SIMILAR PROPERTIES PAGE
# ------------------------------------------------------------

elif page == "Similar Properties":

    st.header("Similar Properties")
    st.write(
        "Find comparable listings from the saved training reference data."
    )

    selected_location = st.selectbox(
        "Location",
        known_localities,
    )

    selected_bhk = st.number_input(
        "BHK",
        min_value=1,
        max_value=15,
        value=2,
        step=1,
    )

    selected_area = st.number_input(
        "Area (sq. ft.)",
        min_value=100.0,
        max_value=50000.0,
        value=1200.0,
        step=50.0,
    )

    if st.button("Find Similar Properties"):
        similar = reference_df[
            (reference_df["location"] == selected_location)
            & (np.isclose(reference_df["bhk"], selected_bhk))
        ].copy()

        if similar.empty:
            st.warning(
                "No comparable listings found for this locality and BHK."
            )
        else:
            similar["area_difference"] = (
                similar["total_sqft"] - selected_area
            ).abs()

            similar = similar.sort_values("area_difference").head(10)

            display_columns = [
                "location",
                "area_type",
                "total_sqft",
                "bhk",
                "bath",
                "balcony",
                "price",
            ]

            st.dataframe(
                similar[display_columns],
                use_container_width=True,
                hide_index=True,
            )


# ------------------------------------------------------------
# 11. EXPLAINABLE AI PAGE — SHAP
# ------------------------------------------------------------

elif page == "Explainable AI":

    st.header("Explainable AI — SHAP Analysis")
    st.write(
        "Understand how the model's input features contribute to its "
        "predictions. SHAP explains model behavior; it does not establish "
        "causation."
    )

    if st.button("Generate SHAP Analysis"):
        try:
            import matplotlib.pyplot as plt
            import shap

            shap_reference = load_shap_reference()

            missing_columns = [
                col for col in FEATURE_COLUMNS
                if col not in shap_reference.columns
            ]

            if missing_columns:
                st.error(
                    "shap_reference.csv is missing required model features: "
                    + ", ".join(missing_columns)
                )
                st.stop()

            if shap_reference.empty:
                st.error("shap_reference.csv contains no rows.")
                st.stop()

            X_explain = shap_reference[FEATURE_COLUMNS].sample(
                n=min(100, len(shap_reference)),
                random_state=42,
            ).copy()

            for column in CATEGORICAL_FEATURES:
                X_explain[column] = (
                    X_explain[column]
                    .fillna("__MISSING__")
                    .astype(str)
                )

            st.subheader("Global Feature Importance")

            with st.spinner("Calculating SHAP values..."):
                explainer = shap.TreeExplainer(model)
                shap_values = explainer(X_explain)

            # Bar plot
            plt.close("all")
            shap.plots.bar(
                shap_values,
                max_display=10,
                show=False,
            )
            bar_fig = plt.gcf()
            st.pyplot(bar_fig, clear_figure=True)
            plt.close(bar_fig)

            # Beeswarm plot
            st.subheader("Feature Impact Distribution")
            plt.close("all")
            shap.plots.beeswarm(
                shap_values,
                max_display=10,
                show=False,
            )
            beeswarm_fig = plt.gcf()
            st.pyplot(beeswarm_fig, clear_figure=True)
            plt.close(beeswarm_fig)

            st.success(
                "SHAP analysis completed using the saved model-feature "
                "reference dataset."
            )

        except ImportError:
            st.error(
                "SHAP or Matplotlib is not installed. Install both packages "
                "using: pip install shap matplotlib"
            )
        except Exception as error:
            st.error(f"SHAP analysis failed: {error}")


# ------------------------------------------------------------
# 12. ABOUT PAGE
# ------------------------------------------------------------

elif page == "About GeoValue AI":

    st.header("About GeoValue AI")

    st.markdown(
        """
        **GeoValue AI** is an academic machine-learning project for
        residential property listing-price estimation and real-estate
        listing analytics.

        ### Core technologies

        - Python
        - Pandas and NumPy
        - CatBoost
        - Scikit-learn (model development and evaluation)
        - SHAP
        - Plotly
        - Streamlit

        ### Main capabilities

        1. House price prediction
        2. Locality-level listing analytics
        3. Similar property comparison
        4. Explainable machine learning
        5. Prediction uncertainty estimation

        ### Dataset scope

        The initial dataset represents Bengaluru residential property
        listings. Predictions should not be interpreted as verified
        transaction prices or as a valuation system for cities outside
        the dataset's scope.
        """
    )

    st.info(
        "GeoValue AI is an academic prototype, not a substitute "
        "for professional property valuation."
    )


# ------------------------------------------------------------
# 13. FOOTER
# ------------------------------------------------------------

st.divider()
st.caption(
    "GeoValue AI | Intelligent House Price Prediction "
    "and Real Estate Analytics System"
)
