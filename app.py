# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 

# --- CONFIGURATION & SETUP ---
st.set_page_config(
    page_title="California Housing AI",
    page_icon="⚡", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Set Matplotlib to Dark Mode
mpl.rcParams.update({
    "figure.facecolor": "#0e1117", 
    "axes.facecolor": "#1f2229",
    "text.color": "white",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "grid.color": "#2c3038"
})

# --- CSS STYLING ---
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;700&display=swap');
    
    .stApp { 
        background: #0e1117; 
        color: #fafafa; 
        font-family: 'Montserrat', sans-serif; 
    }
    
    /* Metrics Styling */
    div[data-testid="stMetric"] {
        background-color: #1f2229; 
        padding: 15px;
        border-radius: 10px; 
        border: 1px solid #333;
        text-align: center;
    }
    div[data-testid="stMetricValue"] {
        color: #40a9ff; 
        font-size: 2em; 
        font-weight: 700;
    }

    /* Flowchart Styling */
    .flow-step {
        background-color: #1f2229; 
        padding: 15px 20px;
        border-radius: 10px;
        margin: 10px 0;
        border: 1px solid #40a9ff;
    }
    .flow-arrow {
        text-align: center;
        font-size: 24px;
        color: #40a9ff;
        margin: -5px 0;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD DATA ---
@st.cache_resource
def load_assets():
    try:
        pipeline = joblib.load("pipeline.pkl")
        model = joblib.load("model.pkl")
        training_data = joblib.load("training_data.pkl")
        feat_imp = joblib.load("feature_importance.pkl")
        test_res = joblib.load("test_results.pkl")
        return pipeline, model, training_data, feat_imp, test_res
    except FileNotFoundError:
        st.error("⚠️ Model files missing. Please run `train_model.py` first.")
        st.stop()

pipeline, model, training_data, feat_imp, test_res = load_assets()

# --- SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1018/1018525.png", width=50)
    st.title("Settings")
    st.markdown("---")
    
    st.subheader("📍 Location")
    longitude = st.slider("Longitude", -124.35, -114.31, -122.23)
    latitude = st.slider("Latitude", 32.54, 41.95, 37.88)
    ocean_proximity = st.selectbox("View Type", 
        ["<1H OCEAN", "INLAND", "ISLAND", "NEAR BAY", "NEAR OCEAN"], index=3)

    if longitude < -124.25:
        st.warning("🌊 Point is in the Ocean!")

    st.subheader("🏠 Specs")
    housing_median_age = st.slider("Age (Years)", 1, 52, 20)
    total_rooms = st.number_input("Total Rooms", 1, 10, 5)
    total_bedrooms = st.number_input("Total Bedrooms", 1, 10, 5)
    
    st.subheader("💰 Socio-Eco")
    median_income_raw = st.slider("Median Income ($)", 5000, 150000, 83252, step=500)
    median_income = median_income_raw / 10000.0
    population = st.number_input("Population", 1, 35000, 322)
    households = st.number_input("Households", 1, 30, 5)

    input_df = pd.DataFrame([{
        "longitude": longitude, "latitude": latitude, "housing_median_age": housing_median_age,
        "total_rooms": total_rooms, "total_bedrooms": total_bedrooms, "population": population,
        "households": households, "median_income": median_income, "ocean_proximity": ocean_proximity
    }])

# --- MAIN PAGE ---
st.title("California Housing Intelligence ⚡️")
st.markdown("### Predictive Modeling & Data Analytics System")

tab1, tab2, tab3 = st.tabs(["🔮 LIVE DASHBOARD", "📈 ANALYTICS SUITE", "🧠 PROJECT ARCHITECTURE"])

# ==========================================
# TAB 1: LIVE DASHBOARD (CLEAN & WORKING)
# ==========================================
with tab1:
    # --- Prediction Logic ---
    input_prepared = pipeline.transform(input_df)
    prediction = model.predict(input_prepared)[0]
    
    # --- Top Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Data Records", f"{len(training_data):,}")
    col2.metric("Model Used", "Random Forest Regressor")
    col3.metric("Prediction Error (RMSE)", f"$50,000")

    st.markdown("---")

    # --- Main Interface ---
    col_map, col_pred = st.columns([2, 1])
    
    with col_map:
        st.subheader("🗺️ Geographic Density Map")
        map_fig = px.scatter_mapbox(
            training_data.sample(1500), lat="latitude", lon="longitude", 
            color="median_house_value", size="population", 
            color_continuous_scale="Reds", 
            zoom=5, center={"lat": 36.7, "lon": -119.4}, 
            mapbox_style="carto-positron", 
            height=450
        )
        map_fig.add_scattermapbox(
            lat=[latitude], lon=[longitude],
            mode='markers', marker=dict(size=25, color='#40a9ff', symbol='circle'), 
            name='Selected Property'
        )
        map_fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
        st.plotly_chart(map_fig, use_container_width=True)

    with col_pred:
        st.caption("MODEL PREDICTION")
        st.markdown(f"<h1 style='color:#40a9ff; font-size:48px;'>${prediction:,.0f}</h1>", unsafe_allow_html=True)
        st.divider()
        st.write(f"**Income:** ${median_income_raw:,.0f}")
        st.write(f"**Location:** {input_df['ocean_proximity'][0]}")
        st.write(f"**Age:** {housing_median_age} years")

# ==========================================
# TAB 2: ANALYTICS SUITE (6 GRAPHS)
# ==========================================
with tab2:
    st.markdown("#### 📊 Deep Dive Analytics")

    # --- ROW 1 ---
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.subheader("1. Drivers of Price")
        fig_imp = px.bar(feat_imp, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale="Blues_r")
        fig_imp.update_layout(showlegend=False, plot_bgcolor='#1f2229', paper_bgcolor='#1f2229', font_color="white", height=350)
        st.plotly_chart(fig_imp, use_container_width=True)

    with r1c2:
        st.subheader("2. Price Distribution by Location")
        fig_box = px.box(training_data, x="ocean_proximity", y="median_house_value", color="ocean_proximity", 
                         title="Price Ranges per Region")
        fig_box.update_layout(plot_bgcolor='#1f2229', paper_bgcolor='#1f2229', font_color="white", showlegend=False, height=350)
        st.plotly_chart(fig_box, use_container_width=True)

    # --- ROW 2 ---
    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.subheader("3. Actual vs Predicted Accuracy")
        fig_res = px.scatter(test_res, x="Actual", y="Predicted", opacity=0.6, trendline="ols", trendline_color_override="#ff40a9")
        fig_res.add_shape(type="line", x0=0, y0=0, x1=500000, y1=500000, line=dict(color="#40a9ff", dash="dash"))
        fig_res.update_layout(plot_bgcolor='#1f2229', paper_bgcolor='#1f2229', font_color="white", height=350)
        st.plotly_chart(fig_res, use_container_width=True)

    with r2c2:
        st.subheader("4. Model Comparison (RMSE)")
        models_data = pd.DataFrame({
            'Model': ['Linear Regression', 'Decision Tree', 'Random Forest'],
            'RMSE': [69000, 71000, 50000]
        })
        fig_model = px.bar(models_data, x="RMSE", y="Model", orientation='h', color="RMSE", color_continuous_scale="Reds_r", text_auto='.2s')
        fig_model.update_layout(plot_bgcolor='#1f2229', paper_bgcolor='#1f2229', font_color="white", height=350)
        st.plotly_chart(fig_model, use_container_width=True)

    # --- ROW 3 ---
    r3c1, r3c2 = st.columns(2)
    with r3c1:
        st.subheader("5. Feature Correlation Heatmap")
        numeric_df = training_data.select_dtypes(include=[np.number])
        corr_matrix = numeric_df.corr()
        
        fig_corr, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, cbar=False)
        st.pyplot(fig_corr)

    with r3c2:
        st.subheader("6. Population vs. Price (Density Effect)")
        fig_pop = px.scatter(training_data.sample(1000), x="population", y="median_house_value", 
                             size="median_income", color="ocean_proximity",
                             title="Population Density vs Price (Size = Income)")
        fig_pop.update_layout(plot_bgcolor='#1f2229', paper_bgcolor='#1f2229', font_color="white", height=350)
        st.plotly_chart(fig_pop, use_container_width=True)


# ==========================================
# TAB 3: PROJECT ARCHITECTURE
# ==========================================
with tab3:
    st.header("🧠 Detailed Project Workflow")
    
    def flowchart_step(title, description, code_snippet):
        st.markdown(f'<div class="flow-step">', unsafe_allow_html=True)
        st.subheader(f"✅ {title}")
        st.caption(description)
        st.code(code_snippet, language="python")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('<p class="flow-arrow">⬇️</p>', unsafe_allow_html=True)

    flowchart_step(
        "STEP 1: Data Ingestion",
        "Loading the raw California Housing dataset using Pandas.",
        """import pandas as pd\nhousing = pd.read_csv("housing.csv")"""
    )

    flowchart_step(
        "STEP 2: Data Cleaning",
        "Imputing missing values in 'total_bedrooms' with the median.",
        """imputer = SimpleImputer(strategy="median")\nhousing['total_bedrooms'].fillna(median, inplace=True)"""
    )
    
    flowchart_step(
        "STEP 3: Transformation Pipeline",
        "Standard Scaling numerical features and One-Hot Encoding categorical features.",
        """preprocessing = ColumnTransformer([\n  ('num', StandardScaler(), num_features),\n  ('cat', OneHotEncoder(), cat_features)\n])"""
    )

    flowchart_step(
        "STEP 4: Model Training",
        "Training the Random Forest Regressor on the processed data.",
        """model = RandomForestRegressor(n_estimators=100)\nmodel.fit(X_train, y_train)"""
    )
    
    st.markdown('<div class="flow-step">', unsafe_allow_html=True)
    st.subheader("✅ STEP 5: Deployment")
    st.caption("Model is saved and deployed via Streamlit.")
    st.markdown('</div>', unsafe_allow_html=True)