import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load("pipeline.pkl")

model = load_model()

# Title
st.title("🛒 Sales Forecasting App")

# Sidebar inputs
st.sidebar.header("📥 Input Features")

profit = st.sidebar.number_input("Profit", value=100.0)
discount_pct = st.sidebar.slider("Discount Percentage", 0.0, 1.0, 0.1)
discount = st.sidebar.number_input("Discount", value=10.0)
order_day = st.sidebar.number_input("Order Day", 1, 31, 15)
order_month = st.sidebar.selectbox("Order Month", list(range(1, 13)))
quantity = st.sidebar.number_input("Quantity", value=5)
operating_expenses = st.sidebar.number_input("Operating Expenses", value=50.0)
ship_day = st.sidebar.number_input("Ship Day", 1, 31, 17)
order_weekday = st.sidebar.selectbox("Order Weekday", 
    ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])

weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, 
               "Thursday": 3, "Friday": 4, "Saturday": 5, "Sunday": 6}
order_weekday_encoded = weekday_map[order_weekday]

# Create input DataFrame
input_df = pd.DataFrame([{
    'Profit': profit,
    'Discount Percentage': discount_pct,
    'Discount': discount,
    'Order Day': order_day,
    'Order Month': order_month,
    'Quantity': quantity,
    'Operating Expenses': operating_expenses,
    'Ship Day': ship_day,
    'Order Weekday': order_weekday_encoded
}])

# Predict button
if st.button("Predict Sales"):
    prediction = model.predict(input_df)[0]
    st.success(f"💰 Predicted Sales: ${prediction:,.2f}")

# Section for visualizations
st.subheader("📊 Sales Data Visualizations")

# Upload dataset
uploaded_file = st.file_uploader("Upload historical sales data (CSV)", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=["Order Date"])

    if "Sales" in df.columns and "Profit" in df.columns:
        # 1. Sales over time
        st.write("### 📈 Sales Over Time")
        sales_time = df.sort_values("Order Date")
        fig1, ax1 = plt.subplots()
        ax1.plot(sales_time["Order Date"], sales_time["Sales"], color='green')
        ax1.set_title("Sales Over Time")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Sales")
        st.pyplot(fig1)

        # 2. Sales vs Profit
        st.write("### 💹 Sales vs Profit")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=df, x="Profit", y="Sales", ax=ax2)
        ax2.set_title("Sales vs Profit")
        st.pyplot(fig2)

        # 3. Sales by Category or Region
        if "Category" in df.columns:
            st.write("### 🧾 Total Sales by Category")
            cat_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)
            st.bar_chart(cat_sales)

        elif "Region" in df.columns:
            st.write("### 🌍 Total Sales by Region")
            region_sales = df.groupby("Region")["Sales"].sum().sort_values(ascending=False)
            st.bar_chart(region_sales)

        # 4. Correlation Heatmap
        st.write("### 🧪 Correlation Heatmap")
        corr = df.select_dtypes(include=np.number).corr()
        fig4, ax4 = plt.subplots(figsize=(10, 6))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax4)
        st.pyplot(fig4)

        # 5. Feature Importance
        st.write("### 🧠 Feature Importance")
        try:
            feature_names = input_df.columns
            importances = model.named_steps['regressor'].feature_importances_
            fi_df = pd.DataFrame({
                "Feature": feature_names,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            fig5, ax5 = plt.subplots()
            sns.barplot(data=fi_df, x="Importance", y="Feature", ax=ax5)
            ax5.set_title("Feature Importance")
            st.pyplot(fig5)
        except Exception:
            st.info("Feature importance not available for this model.")
