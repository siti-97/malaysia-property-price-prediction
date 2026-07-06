%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Force wide mode to utilize screen real estate properly
st.set_page_config(page_title="Malaysian Property Predictor", layout="wide")

@st.cache_resource
def load_assets():
    model = joblib.load('lgbm_model.pkl')
    scaler = joblib.load('robust_scaler.pkl')
    df = pd.read_csv('dashboard_data.csv')
    return model, scaler, df

model, scaler, df = load_assets()

# --- SIDEBAR: INTERACTIVE INPUTS [Fulfills Q4(a)(ii)] ---
with st.sidebar:
    st.header("🏢 Property Specifications")
    
    # 1. Categorical Dropdowns
    selected_state = st.selectbox("Select State", df['state'].unique())
    selected_type = st.selectbox("Property Type", df['property_type'].unique())
    
    st.markdown("---") 
    
    # 2. Numerical Inputs/Sliders
    col_input1, col_input2 = st.columns(2)
    with col_input1:
        built_up = st.number_input("Built-up Area (sqft)", min_value=300, max_value=10000, value=1200, step=50)
        beds = st.number_input("Bedrooms", min_value=1, max_value=10, value=3)
    with col_input2:
        amenities = st.number_input("Amenities (3km)", min_value=0, max_value=20, value=5)
        baths = st.number_input("Bathrooms", min_value=1, max_value=8, value=2)
        
    car_parks = st.slider("Car Parks", 0, 5, 2)
    
    st.markdown("---")
    predict_btn = st.button("🔮 Calculate Estimated Value", use_container_width=True)

# --- MAIN PANEL: PREDICTIVE OUTPUT [Fulfills Q4(a)(iii)] ---
st.title("🇲🇾 Malaysian Property Price Analytics")
st.caption("Agile data product utilizing LightGBM machine learning to estimate real estate valuation.")

if predict_btn:
    st.markdown("### 🎯 Valuation Result")
    
    # Logic: Group rare property types to match training data
    top_5_types = df['property_type'].value_counts().head(5).index
    type_grouped = selected_type if selected_type in top_5_types else 'Other'
    
    input_data = pd.DataFrame({
        'built_up_area_numeric': [built_up],
        'bedroom_numeric': [beds],
        'bathroom_numeric': [baths],
        'car_parks': [car_parks],
        'total_amenities_3km': [amenities],
        'state': [selected_state],
        'property_type_grouped': [type_grouped]
    })
    
    # Scale numeric features
    num_cols = ['built_up_area_numeric', 'bedroom_numeric', 'bathroom_numeric', 'car_parks', 'total_amenities_3km']
    input_data[[f"{c}_scaled" for c in num_cols]] = scaler.transform(input_data[num_cols])
    
    # One-hot encode and align columns
    encoded_input = pd.get_dummies(input_data, columns=['state', 'property_type_grouped'])
    for col in model.feature_name_:
        if col not in encoded_input.columns:
            encoded_input[col] = 0
            
    # Predict and reverse Log transform
    log_pred = model.predict(encoded_input[model.feature_name_])[0]
    actual_price = np.expm1(log_pred)

    # Display Result
    col_metric, col_model = st.columns([3, 1])
    with col_metric:
        st.metric(label="Estimated Market Value", value=f"RM {actual_price:,.2f}")
    with col_model:
        st.info("Model: LightGBM Regressor")
else:
    st.info("💡 Adjust the properties in the sidebar and click 'Calculate Estimated Value' to run the prediction model.")

st.markdown("---")

# --- VISUALIZATIONS [Fulfills Q4(a)(i)] ---
st.subheader("📊 Market Insights")
tab1, tab2, tab3 = st.tabs(["📈 Price Distribution", "🏛️ Top States Average", "📐 Built-up vs Value"])

with tab1:
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    sns.histplot(np.log1p(df['price_numeric']), bins=40, kde=True, color='teal', ax=ax1)
    ax1.set_title("Market Price Distribution (Log Scale)")
    ax1.set_xlabel("Log Price")
    st.pyplot(fig1)
    
with tab2:
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    top_states = df.groupby('state')['price_numeric'].median().sort_values(ascending=False).head(5)
    sns.barplot(x=top_states.values, y=top_states.index, palette='viridis', ax=ax2)
    ax2.set_title("Median Price by Top States")
    ax2.set_xlabel("Median Price (RM)")
    ax2.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000000:.1f}M'))
    st.pyplot(fig2)
    
with tab3:
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    sample_df = df.sample(n=1000, random_state=42) 
    sns.scatterplot(x='built_up_area_numeric', y='price_numeric', data=sample_df, alpha=0.5, color='coral', ax=ax3)
    ax3.set_title("Built-up Area vs Property Value")
    ax3.set_xlim(0, 5000)
    ax3.set_ylim(0, 3000000)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000000:.1f}M'))
    st.pyplot(fig3)
