# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="🏠 Real Estate App", layout="wide")

# ---------------- APP TITLE ----------------
st.markdown("<h1 style='text-align:center;'>🏠 Real Estate Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Use the sidebar to navigate and filter properties.</p>", unsafe_allow_html=True)

# ------------------ LOAD DATA ------------------
@st.cache_data
def load_data():
    df = pd.read_csv("my_data.csv")
    # Basic cleaning for safety
    df.columns = df.columns.str.strip()
    df.dropna(how="all", inplace=True)
    return df

df = load_data()

# ------------------ SIDEBAR NAVIGATION ------------------
st.sidebar.title("🏠 Real Estate Dashboard")
st.sidebar.markdown("---")

# Navigation Menu in Sidebar
menu = st.sidebar.radio(
    "📍 Navigate to:",
    ["Dashboard", "Properties", "Analysis", "Prediction"]
)

st.sidebar.markdown("---")

# ---------------- PAGE ROUTING ----------------
if menu == "Dashboard":
    # ------------------ MAIN PAGE FILTERS (NOT SIDEBAR) ------------------
    st.markdown("### 🎯 Filter Your Data")
    col1, col2 = st.columns(2)
    
    if "price" in df.columns:
        min_price, max_price = float(df["price"].min()), float(df["price"].max())
        price_range = col1.slider("Select Price Range (₹)", min_price, max_price, (min_price, max_price))
    
    if "area_sqft" in df.columns:
        min_area, max_area = float(df["area_sqft"].min()), float(df["area_sqft"].max())
        area_range = col2.slider("Select Area Range (sqft)", min_area, max_area, (min_area, max_area))
    
    # Apply filters
    if "price" in df.columns:
        df = df[(df["price"] >= price_range[0]) & (df["price"] <= price_range[1])]
    
    if "area_sqft" in df.columns:
        df = df[(df["area_sqft"] >= area_range[0]) & (df["area_sqft"] <= area_range[1])]
    
    # ------------------ METRICS SECTION ------------------
    st.markdown("### 📊 Key Property Insights")
    
    total_properties = len(df)
    avg_price = df["price"].mean() if "price" in df.columns else 0
    avg_area = df["area_sqft"].mean() if "area_sqft" in df.columns else 0
    avg_rating = df["rating"].mean() if "rating" in df.columns else 0
    top_society = df["society"].mode()[0] if "society" in df.columns and not df["society"].isna().all() else "N/A"
    
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("🏘️ Total Properties", f"{total_properties}")
    col2.metric("💰 Avg. Price", f"₹ {avg_price:,.0f}")
    col3.metric("📐 Avg. Area", f"{avg_area:,.0f} sqft")
    col4.metric("⭐ Avg. Rating", f"{avg_rating:.1f}" if avg_rating > 0 else "N/A")
    col5.metric("🏙️ Top Society", top_society)
    
    st.markdown("---")
    
    # ------------------ ADDITIONAL INSIGHTS ------------------
    st.markdown("### 🏢 Society Insights")
    
    if "society" in df.columns:
        top_5_societies = df["society"].value_counts().head(5)
        fig, ax = plt.subplots(figsize=(6, 3))
        bars = ax.barh(top_5_societies.index[::-1], top_5_societies.values[::-1], color="skyblue", edgecolor="black")
        ax.set_title("Top 5 Most Listed Societies", fontsize=12)
        ax.set_xlabel("Number of Listings", fontsize=10)
        ax.tick_params(axis='y', labelsize=9)
        ax.tick_params(axis='x', labelsize=9)
        st.pyplot(fig)
    
    if "price" in df.columns and "bedRoom" in df.columns:
        avg_price_by_bhk = df.groupby("bedRoom")["price"].mean().sort_index()
        fig2, ax2 = plt.subplots(figsize=(6, 3))
        ax2.plot(avg_price_by_bhk.index, avg_price_by_bhk.values, marker='o', color='teal')
        ax2.set_title("Average Price by Bedrooms (BHK)", fontsize=12)
        ax2.set_xlabel("Bedrooms", fontsize=10)
        ax2.set_ylabel("Avg. Price (₹)", fontsize=10)
        ax2.tick_params(axis='both', labelsize=9)
        st.pyplot(fig2)
    
    st.markdown("---")
    
    # ------------------ VISUALIZATIONS ------------------
    st.markdown("### 📈 Data Visualizations")
    
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Price Distribution", "Area Distribution", "Price vs Area", "Map view", "Furnishing", "Rating Breakdown", "Property ROI & Price Projection"])
    
    with tab1:
        st.subheader("💰 Price Distribution")
        if "price" in df.columns:
            filtered_price = df["price"][df["price"] <= df["price"].quantile(0.99)]
            
            mean_price = filtered_price.mean()
            median_price = filtered_price.median()
            
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.histplot(filtered_price, bins=30, kde=True, color="skyblue", edgecolor="black", ax=ax2)
            ax2.axvline(mean_price, color="red", linestyle="--", linewidth=2, label=f"Mean: ₹{mean_price:,.0f}")
            ax2.axvline(median_price, color="green", linestyle="-.", linewidth=2, label=f"Median: ₹{median_price:,.0f}")
            ax2.scatter([mean_price, median_price], [0, 0], color=["red", "green"], s=60, zorder=5)
            ax2.text(mean_price, ax2.get_ylim()[1]*0.9, 'Mean', color='red', fontsize=8, ha='center')
            ax2.text(median_price, ax2.get_ylim()[1]*0.8, 'Median', color='green', fontsize=8, ha='center')
            ax2.set_title("💰 Price Distribution (with Mean & Median)", fontsize=12, pad=10)
            ax2.set_xlabel("Price (₹)", fontsize=10)
            ax2.set_ylabel("Count", fontsize=10)
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend(fontsize=8)
            ax2.set_xlim(0, filtered_price.quantile(0.95))
            ax2.tick_params(axis='both', labelsize=8)
            plt.tight_layout()
            st.pyplot(fig2)
    
    with tab2:
        st.subheader("📏 Area Distribution")
        if "area_sqft" in df.columns:
            filtered_area = df["area_sqft"][df["area_sqft"] <= df["area_sqft"].quantile(0.99)]
            fig2, ax2 = plt.subplots(figsize=(7, 4))
            ax2.hist(filtered_area, bins=30, edgecolor="black", color="mediumseagreen")
            ax2.set_title("Area Distribution (Excluding Outliers)", fontsize=12)
            ax2.set_xlabel("Area (sqft)", fontsize=10)
            ax2.set_ylabel("Count", fontsize=10)
            ax2.tick_params(axis='both', labelsize=8)
            st.pyplot(fig2)
    
    with tab3:
        st.subheader("💹 Price vs Area (Colored by Society)")
        if "price" in df.columns and "area_sqft" in df.columns:
            df["price"] = pd.to_numeric(df["price"], errors="coerce")
            df["area_sqft"] = pd.to_numeric(df["area_sqft"], errors="coerce")
            df_filtered = df.dropna(subset=["price", "area_sqft"])
            df_filtered = df_filtered[
                (df_filtered["price"] <= df_filtered["price"].quantile(0.99)) &
                (df_filtered["area_sqft"] <= df_filtered["area_sqft"].quantile(0.99))
            ]
            
            if not df_filtered.empty:
                try:
                    fig = px.scatter(
                        df_filtered,
                        x="area_sqft",
                        y="price",
                        color="society",
                        hover_data={
                            "society": True,
                            "bedRoom": True if "bedRoom" in df.columns else False,
                            "bathroom": True if "bathroom" in df.columns else False,
                            "price": ":,.0f",
                            "area_sqft": ":,.0f"
                        },
                        title="Price vs Area (Top 99% Data, Colored by Society)"
                    )
                    fig.update_layout(
                        xaxis_title="Area (sqft)",
                        yaxis_title="Price (₹)",
                        legend_title="Society",
                        title_font=dict(size=16),
                        font=dict(size=10),
                        height=500,
                        template="plotly_white",
                        margin=dict(l=40, r=40, t=80, b=40)
                    )
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning("⚠️ Plotly chart failed to render, showing backup chart.")
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(10, 6))
                    plt.scatter(df_filtered["area_sqft"], df_filtered["price"], alpha=0.6)
                    plt.xlabel("Area (sqft)")
                    plt.ylabel("Price (₹)")
                    plt.title("Price vs Area (Colored by Society)")
                    st.pyplot(plt)
            else:
                st.error("No valid data available after filtering.")
        else:
            st.error("Required columns 'price' and 'area_sqft' not found.")
    
    with tab4:
        st.subheader("🗺️ Interactive Map View - Top Properties")
        if {"address", "society", "price"}.issubset(df.columns):
            df_map = df.copy()
            df_map = df_map.dropna(subset=["address"])
            df_map = df_map.sort_values(by="price", ascending=False).head(200)
            df_map["location_info"] = df_map["society"].fillna("") + " | " + df_map["address"].fillna("")
            
            if {"latitude", "longitude"}.issubset(df_map.columns):
                lat_col, lon_col = "latitude", "longitude"
            else:
                np.random.seed(42)
                df_map["latitude"] = np.random.uniform(28.3, 28.6, len(df_map))
                df_map["longitude"] = np.random.uniform(76.8, 77.2, len(df_map))
                lat_col, lon_col = "latitude", "longitude"
            
            fig = px.scatter_mapbox(
                df_map,
                lat=lat_col,
                lon=lon_col,
                hover_name="society",
                hover_data={
                    "price": True,
                    "area_sqft": True,
                    "bedRoom": True,
                    "bathroom": True,
                    "location_info": True,
                },
                color="price",
                size="area_sqft",
                color_continuous_scale="Viridis",
                zoom=10,
                height=600,
            )
            fig.update_layout(
                mapbox_style="carto-positron",
                margin={"r": 0, "t": 30, "l": 0, "b": 0},
                title="📍 Top Properties (Interactive View)",
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("#### 🏠 Top 10 Expensive Properties")
            st.dataframe(
                df_map[["society", "address", "price", "area_sqft", "bedRoom", "bathroom"]]
                .sort_values(by="price", ascending=False)
                .head(10)
                .style.format({"price": "₹{:,.0f}", "area_sqft": "{:,.0f}"}),
                use_container_width=True
            )
        else:
            st.warning("⚠️ Columns 'address', 'society', and 'price' are required to create the map view.")
    
    with tab5:
        st.subheader("🛋️ Furnishing Overview")
        if "furnish_status" in df.columns:
            df["furnish_status"] = df["furnish_status"].fillna("Unknown").str.title()
            furnish_counts = df["furnish_status"].value_counts()
            
            fig1, ax1 = plt.subplots(figsize=(5, 5))
            ax1.pie(
                furnish_counts.values,
                labels=furnish_counts.index,
                autopct='%1.1f%%',
                startangle=90,
                colors=["#66b3ff", "#99ff99", "#ffcc99", "#ff9999"],
                wedgeprops={'edgecolor': 'black'}
            )
            ax1.set_title("Distribution of Furnishing Status", fontsize=13)
            st.pyplot(fig1)
            
            colA, colB, colC = st.columns(3)
            colA.metric("🏠 Total Properties", f"{len(df)}")
            colB.metric("🛋️ Furnished", f"{furnish_counts.get('Furnished', 0)}")
            colC.metric("🚪 Semi/Unfurnished", f"{furnish_counts.get('Semi-Furnished', 0) + furnish_counts.get('Unfurnished', 0)}")
            
            st.markdown("---")
            st.markdown("### 🧱 Furnishing Item Insights")
            
            if "furnishDetails" in df.columns:
                furnish_details_df = df[["furnish_status", "furnishDetails"]].dropna()
                furnishing_items = []
                
                for _, row in furnish_details_df.iterrows():
                    items = [x.strip().strip("'").strip('"') for x in str(row["furnishDetails"]).replace(",", " ").split()]
                    furnishing_items.extend([(row["furnish_status"], i) for i in items if i])
                
                if furnishing_items:
                    furnish_df = pd.DataFrame(furnishing_items, columns=["FurnishStatus", "Item"])
                    top_items = furnish_df["Item"].value_counts().head(10)
                    
                    fig2, ax2 = plt.subplots(figsize=(8, 4))
                    ax2.barh(top_items.index[::-1], top_items.values[::-1], color="mediumseagreen", edgecolor="black")
                    ax2.set_title("Top 10 Most Common Furnishing Items", fontsize=12)
                    ax2.set_xlabel("Frequency", fontsize=10)
                    ax2.tick_params(axis='both', labelsize=9)
                    plt.tight_layout()
                    st.pyplot(fig2)
                    
                    st.markdown("### 🔍 Breakdown by Furnishing Type")
                    for status in furnish_df["FurnishStatus"].unique():
                        st.markdown(f"*{status} Properties:*")
                        sub_items = furnish_df[furnish_df["FurnishStatus"] == status]["Item"].value_counts().head(5)
                        st.write(", ".join(sub_items.index))
                        st.markdown("---")
                    
                    with st.expander("📋 View Detailed Furnishing Data"):
                        st.dataframe(df[["property_name", "society", "furnish_status", "furnishDetails"]].fillna("N/A"))
                else:
                    st.warning("⚠️ No detailed furnishing items found.")
            else:
                st.warning("⚠️ Column 'furnishDetails' not found in dataset.")
        else:
            st.error("❌ Column 'furnish_status' not found in dataset.")
    
    with tab6:
        st.subheader("⭐ Property Ratings & Quality Insights")
        rating_cols = [
            "Environment", "Safety", "Lifestyle", "Connectivity",
            "Green Area", "Amenitie", "Management", "Construction"
        ]
        available_ratings = [col for col in rating_cols if col in df.columns]
        
        if "avg_rating" in df.columns:
            avg_overall = df["avg_rating"].mean()
            st.metric("🌟 Average Overall Rating", f"{avg_overall:.2f} / 5")
        
        st.markdown("---")
        
        if "rating" in df.columns:
            fig1, ax1 = plt.subplots(figsize=(6, 3))
            sns.histplot(df["rating"], bins=10, kde=True, color="goldenrod", edgecolor="black", ax=ax1)
            ax1.set_title("Overall Rating Distribution", fontsize=12)
            ax1.set_xlabel("Rating")
            ax1.set_ylabel("Count")
            ax1.grid(alpha=0.4, linestyle="--")
            st.pyplot(fig1)
        else:
            st.warning("No 'rating' column found in dataset.")
        
        st.markdown("---")
        
        if available_ratings:
            avg_scores = df[available_ratings].mean().sort_values(ascending=False)
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            bars = ax2.bar(avg_scores.index, avg_scores.values, color="seagreen", edgecolor="black", alpha=0.8)
            ax2.set_title("Average Score by Category", fontsize=12)
            ax2.set_ylabel("Average Rating (out of 5)")
            ax2.set_ylim(0, 5)
            ax2.tick_params(axis='x', rotation=30)
            for bar in bars:
                ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         f"{bar.get_height():.2f}", ha='center', fontsize=8)
            st.pyplot(fig2)
        else:
            st.info("No detailed rating columns available for breakdown.")
    
    with tab7:
        st.title("📊 Real Estate Data Analysis and ROI Prediction")
        try:
            df_roi = pd.read_csv("my_data.csv")
            st.success("✅ Data loaded successfully")
        except Exception as e:
            st.error("❌ Error loading dataset.")
            st.stop()
        
        df_roi = df_roi.dropna(subset=["price", "area_sqft"])
        df_roi = df_roi.head(2000)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("🏡 Total Properties", len(df_roi))
        with col2:
            st.metric("📏 Avg Area (sqft)", round(df_roi["area_sqft"].mean(), 2))
        with col3:
            st.metric("💰 Avg Price (₹ Lakh)", round(df_roi["price"].mean(), 2))
        
        st.markdown("### 💹 ROI (Return on Investment) Simulation")
        
        col_roi1, col_roi2 = st.columns(2)
        with col_roi1:
            price_growth = st.slider("Select Annual Price Growth (%)", 1, 20, 7, key="tab7_growth")
        with col_roi2:
            investment_years = st.slider("Investment Duration (Years)", 1, 15, 5, key="tab7_years")
        
        df_roi["roi_percent"] = ((1 + (price_growth / 100)) ** investment_years - 1) * 100
        
        fig_roi = px.histogram(df_roi, x="roi_percent", nbins=30, color_discrete_sequence=["green"],
                               title=f"ROI Distribution ({price_growth}% Growth for {investment_years} Years)")
        st.plotly_chart(fig_roi, use_container_width=True)
        st.write(f"**Average ROI:** {df_roi['roi_percent'].mean():.2f}%")
        
        st.markdown("### 📈 Price vs Area Analysis")
        fig_scatter = px.scatter(df_roi, x="area_sqft", y="price", color="bedRoom",
                                 hover_data=["society", "address"], title="Price vs Area")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
        st.markdown("### 🏢 Top 10 Societies by Average Price")
        top_societies = df_roi.groupby("society")["price"].mean().nlargest(10).reset_index()
        fig_soc = px.bar(top_societies, x="society", y="price", color="price",
                         title="Top 10 Expensive Societies")
        st.plotly_chart(fig_soc, use_container_width=True)
        
        st.markdown("### 🔥 Correlation Heatmap")
        numeric_cols = df_roi.select_dtypes(include=[np.number])
        corr = numeric_cols.corr()
        fig_corr, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
        st.pyplot(fig_corr)
        
        st.markdown("### 🤖 ROI Prediction using Linear Regression")
        feature_cols = ["price", "area_sqft", "bedRoom", "bathroom", "balcony"]
        target_col = "roi_percent"
        df_roi = df_roi.fillna(0)
        X = df_roi[feature_cols]
        y = df_roi[target_col]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("📉 Mean Absolute Error", round(mae, 2))
        with col2:
            st.metric("📈 R² Score", round(r2, 2))
        
        fig_pred = px.scatter(x=y_test, y=y_pred, labels={'x': "Actual ROI", 'y': "Predicted ROI"},
                              title="Actual vs Predicted ROI")
        st.plotly_chart(fig_pred, use_container_width=True)
        
        st.markdown("### ⚠️ Investment Precaution Suggestions")
        
        def precaution(row):
            if row["roi_percent"] > 70:
                return "🚀 Excellent Investment"
            elif row["roi_percent"] > 40:
                return "👍 Safe Option"
            elif row["roi_percent"] > 20:
                return "⚠️ Moderate Risk"
            else:
                return "❌ Not Recommended"
        
        df_roi["Precaution"] = df_roi.apply(precaution, axis=1)
        st.dataframe(df_roi[["property_name", "society", "price", "roi_percent", "Precaution"]].head(15))
        
        st.download_button(
            label="⬇️ Download Analysis Report (CSV)",
            data=df_roi.to_csv(index=False).encode("utf-8"),
            file_name="RealEstate_Analysis_Report.csv",
            mime="text/csv"
        )

elif menu == "Properties":
    # ------------------ MAIN PAGE FILTERS (NOT SIDEBAR) ------------------
    st.markdown("### 🔍 Filter Your Property")
    
    # Simulate missing lat/lon if not present
    if "latitude" not in df.columns or "longitude" not in df.columns:
        np.random.seed(42)
        df["latitude"] = np.random.uniform(28.3, 28.7, len(df))
        df["longitude"] = np.random.uniform(76.8, 77.3, len(df))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        min_price, max_price = int(df["price"].min()), int(df["price"].max())
        price_range = st.slider("💰 Price Range (in ₹)", min_price, max_price, (min_price, max_price), key="prop_price")
    
    with col2:
        min_area, max_area = int(df["area_sqft"].min()), int(df["area_sqft"].max())
        area_range = st.slider("📏 Area (sqft)", min_area, max_area, (min_area, max_area), key="prop_area")
    
    with col3:
        bedrooms = st.multiselect("🛏️ Bedrooms", sorted(df["bedRoom"].dropna().unique()))
    
    col4, col5, col6 = st.columns(3)
    with col4:
        furnishing = st.multiselect("🪑 Furnishing Status", sorted(df["furnish_status"].dropna().unique()))
    
    with col5:
        green_range = st.slider(
            "🌿 Green / Environment Rating",
            float(df["Green Area"].min()),
            float(df["Green Area"].max()),
            (float(df["Green Area"].min()), float(df["Green Area"].max())),
            key="prop_green"
        )
    
    with col6:
        safety_range = st.slider(
            "🛡️ Safety Rating",
            float(df["Safety"].min()),
            float(df["Safety"].max()),
            (float(df["Safety"].min()), float(df["Safety"].max())),
            key="prop_safety"
        )
    
    connectivity_range = st.slider(
        "🚆 Connectivity Rating",
        float(df["Connectivity"].min()),
        float(df["Connectivity"].max()),
        (float(df["Connectivity"].min()), float(df["Connectivity"].max())),
        key="prop_connect"
    )
    
    # ----------------- Apply Filters -----------------
    filtered_df = df[
        (df["price"].between(price_range[0], price_range[1])) &
        (df["area_sqft"].between(area_range[0], area_range[1])) &
        (df["Green Area"].between(green_range[0], green_range[1])) &
        (df["Safety"].between(safety_range[0], safety_range[1])) &
        (df["Connectivity"].between(connectivity_range[0], connectivity_range[1]))
    ]
    
    if bedrooms:
        filtered_df = filtered_df[filtered_df["bedRoom"].isin(bedrooms)]
    
    if furnishing:
        filtered_df = filtered_df[filtered_df["furnish_status"].isin(furnishing)]
    
    filtered_df = filtered_df.sort_values(by="price", ascending=False).head(2000)
    
    st.markdown(f"### 🏘️ Showing {len(filtered_df)} matching properties")
    
    # ----------------- Map Visualization -----------------
    if not filtered_df.empty:
        fig = px.scatter_mapbox(
            filtered_df,
            lat="latitude",
            lon="longitude",
            color="price",
            size="area_sqft",
            hover_name="property_name",
            hover_data={
                "society": True,
                "address": True,
                "bedRoom": True,
                "bathroom": True,
                "furnish_status": True,
                "price": True,
                "area_sqft": True,
            },
            color_continuous_scale="Viridis",
            zoom=10,
            height=600,
        )
        
        fig.update_layout(
            mapbox_style="carto-positron",
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            title="📍 Filtered Property Locations"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 🏡 Property Details")
        st.dataframe(
            filtered_df[
                ["property_name", "society", "address", "price", "area_sqft", "bedRoom",
                 "bathroom", "furnish_status", "Green Area", "Safety", "Connectivity", "avg_rating"]
            ].style.format({
                "price": "₹{:,.0f}",
                "area_sqft": "{:,.0f}",
                "avg_rating": "{:.1f}"
            }),
            use_container_width=True
        )
    else:
        st.warning("⚠️ No properties found matching the selected criteria.")

elif menu == "Analysis":
    st.title("📊 Real Estate Data Analysis & Insights")
    st.markdown("Analyze trends, returns, and property performance metrics below.")
    
    if "latitude" not in df.columns or "longitude" not in df.columns:
        np.random.seed(42)
        df["latitude"] = np.random.uniform(28.3, 28.7, len(df))
        df["longitude"] = np.random.uniform(76.8, 77.3, len(df))
    
    st.subheader("📈 Statistical Summary")
    st.dataframe(df.describe().T, use_container_width=True)
    
    st.subheader("💰 Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["price"], bins=50, kde=True, color="royalblue", ax=ax1)
    ax1.set_xlabel("Price (₹)")
    ax1.set_ylabel("Properties Count")
    st.pyplot(fig1)
    
    st.subheader("📏 Area vs Price Relationship")
    fig2 = px.scatter(df, x="area_sqft", y="price", color="bedRoom",
                      hover_data=["society", "address"], title="Area vs Price by Bedrooms")
    st.plotly_chart(fig2, use_container_width=True)
    
    st.subheader("🪑 Furnishing Type vs Price per Sqft")
    if "furnish_status" in df.columns:
        df["rate_per_sqft"] = df["price"] / df["area_sqft"]
        furnish_avg = df.groupby("furnish_status")["rate_per_sqft"].mean().sort_values(ascending=False).reset_index()
        fig3 = px.bar(furnish_avg, x="furnish_status", y="rate_per_sqft", text_auto=True,
                      title="Avg Price per Sqft by Furnishing Type", color="furnish_status")
        st.plotly_chart(fig3, use_container_width=True)
    
    st.subheader("🛏️ Bedroom and Furnishing Distribution")
    fig4 = px.histogram(df, x="bedRoom", color="furnish_status", barmode="group",
                        title="Bedroom Count vs Furnishing Status")
    st.plotly_chart(fig4, use_container_width=True)
    
    st.subheader("🌿 Ratings Impact on Property Value")
    rating_cols = ["Green Area", "Safety", "Connectivity", "Lifestyle", "Amenitie"]
    for col in rating_cols:
        if col in df.columns:
            fig = px.scatter(df, x=col, y="price", trendline="ols",
                             title=f"{col} vs Price", color="bedRoom")
            st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("🛏️ Bedroom Distribution")
    bedroom_counts = df["bedRoom"].value_counts().sort_index()
    fig3, ax3 = plt.subplots()
    sns.barplot(x=bedroom_counts.index, y=bedroom_counts.values, ax=ax3)
    ax3.set_xlabel("Bedrooms")
    ax3.set_ylabel("Number of Properties")
    st.pyplot(fig3)
    
    st.subheader("🪑 Furnishing Status")
    furnish_counts = df["furnish_status"].value_counts()
    fig4, ax4 = plt.subplots()
    sns.barplot(x=furnish_counts.index, y=furnish_counts.values, ax=ax4)
    ax4.set_xlabel("Furnishing Type")
    ax4.set_ylabel("Count")
    st.pyplot(fig4)
    
    st.subheader("🔗 Correlation Heatmap")
    corr_cols = ["price", "area_sqft", "Green Area", "Safety", "Connectivity", "avg_rating"]
    corr = df[corr_cols].corr()
    fig5, ax5 = plt.subplots()
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax5)
    st.pyplot(fig5)
    
    st.subheader("💹 ROI (Return on Investment) Simulation")
    st.markdown("""
    Estimate potential ROI (Return on Investment) based on appreciation and rental yield.
    """)
    
    # Main page filters for ROI (NOT sidebar)
    col_roi1, col_roi2, col_roi3 = st.columns(3)
    with col_roi1:
        avg_growth = st.slider("Select Annual Price Growth (%)", 3, 15, 7, key="analysis_growth")
    with col_roi2:
        years = st.slider("Investment Duration (Years)", 1, 20, 5, key="analysis_years")
    with col_roi3:
        rental_yield = st.slider("Expected Annual Rental Yield (%)", 1, 8, 3, key="analysis_yield")
    
    np.random.seed(42)
    df["growth_rate"] = np.random.normal(avg_growth, 1.5, len(df))
    df["rental_income"] = df["price"] * (rental_yield / 100) * years
    df["expected_future_value"] = df["price"] * ((1 + df["growth_rate"]/100) ** years)
    df["roi_percent"] = ((df["expected_future_value"] + df["rental_income"]) - df["price"]) / df["price"] * 100
    
    fig_roi = px.histogram(
        df, 
        x="roi_percent", 
        nbins=40, 
        color_discrete_sequence=["#00CC96"], 
        title=f"Expected ROI Distribution ({avg_growth}% ± Variation, {rental_yield}% Rental, {years} Years)"
    )
    fig_roi.update_layout(
        xaxis_title="ROI (%)",
        yaxis_title="Count",
        bargap=0.1,
    )
    st.plotly_chart(fig_roi, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Average ROI (%)", f"{df['roi_percent'].mean():.2f}")
    col2.metric("Median ROI (%)", f"{df['roi_percent'].median():.2f}")
    col3.metric("Max ROI (%)", f"{df['roi_percent'].max():.2f}")
    
    st.subheader("🏙️ Top 10 Societies by Average Price")
    if "society" in df.columns:
        top_societies = df.groupby("society")["price"].mean().sort_values(ascending=False).head(10).reset_index()
        fig7 = px.bar(top_societies, x="society", y="price", text_auto=True,
                      title="Top 10 Expensive Societies", color="price")
        st.plotly_chart(fig7, use_container_width=True)
    
    if "Construction" in df.columns and "Management" in df.columns:
        st.subheader("🏗️ Construction & Management Quality vs Avg Rating")
        quality = df.groupby(["Construction", "Management"])["avg_rating"].mean().reset_index()
        fig8 = px.scatter(quality, x="Construction", y="Management",
                          size="avg_rating", color="avg_rating",
                          title="Construction & Management Impact on Ratings")
        st.plotly_chart(fig8, use_container_width=True)
    
    st.subheader("🚨 Outlier Detection (Over/Under Priced Properties)")
    q1, q3 = df["price"].quantile(0.25), df["price"].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    
    outliers = df[(df["price"] > upper) | (df["price"] < lower)]
    st.write(f"Found {len(outliers)} potential outliers.")
    st.dataframe(outliers[["property_name", "society", "price", "area_sqft", "bedRoom"]])
    
    st.success("✅ Analysis Completed Successfully!")

elif menu == "Prediction":
    st.title("🏠 Property Investment ROI & Prediction Dashboard")
    
    st.markdown("""
    Analyze potential *Return on Investment (ROI)* and *Future Price Predictions*
    for each property based on your growth and rental assumptions.
    """)
    
    if "latitude" not in df.columns or "longitude" not in df.columns:
        np.random.seed(42)
        df["latitude"] = np.random.uniform(28.3, 28.7, len(df))
        df["longitude"] = np.random.uniform(76.8, 77.3, len(df))
    
    df = df.dropna(subset=["price", "area_sqft"])
    
    # ------------------ SIDEBAR FILTERS FOR PREDICTION ------------------
    st.sidebar.markdown("### 📈 ROI Simulation Settings")
    
    avg_growth = st.sidebar.slider("🏦 Annual Price Growth (%)", 3, 20, 7, key="pred_growth")
    years = st.sidebar.slider("⏳ Investment Duration (Years)", 1, 25, 5, key="pred_years")
    rental_yield = st.sidebar.slider("🏠 Annual Rental Yield (%)", 1, 10, 3, key="pred_yield")
    
    np.random.seed(42)
    df["growth_rate"] = np.random.normal(avg_growth, 1.5, len(df))
    df["rental_income"] = df["price"] * (rental_yield / 100) * years
    df["predicted_future_price"] = df["price"] * ((1 + df["growth_rate"] / 100) ** years)
    df["predicted_profit"] = (df["predicted_future_price"] + df["rental_income"]) - df["price"]
    df["roi_percent"] = (df["predicted_profit"] / df["price"]) * 100
    
    st.subheader("💹 ROI Distribution Across Properties")
    fig_roi = px.histogram(
        df,
        x="roi_percent",
        nbins=40,
        color_discrete_sequence=["#27AE60"],
        title=f"Expected ROI Distribution ({avg_growth}% Growth, {rental_yield}% Rental, {years} Years)"
    )
    fig_roi.update_layout(
        xaxis_title="ROI (%)",
        yaxis_title="Number of Properties",
        bargap=0.1,
    )
    st.plotly_chart(fig_roi, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("📈 Average ROI (%)", f"{df['roi_percent'].mean():.2f}")
    col2.metric("💰 Max ROI (%)", f"{df['roi_percent'].max():.2f}")
    col3.metric("🏦 Avg Future Value", f"₹{df['predicted_future_price'].mean():,.0f}")
    
    st.subheader("🏘️ Top Societies by Average ROI")
    roi_by_society = (
        df.groupby("society")[["roi_percent", "predicted_profit"]]
        .mean()
        .reset_index()
        .sort_values(by="roi_percent", ascending=False)
        .head(10)
    )
    fig_society = px.bar(
        roi_by_society,
        x="society",
        y="roi_percent",
        text="roi_percent",
        color="roi_percent",
        color_continuous_scale="viridis",
        title="Top 10 Societies by Average ROI (%)"
    )
    fig_society.update_layout(xaxis_title="Society", yaxis_title="Average ROI (%)")
    st.plotly_chart(fig_society, use_container_width=True)
    
    st.subheader("📊 Predicted Property Performance (Top 20 by ROI)")
    st.dataframe(
        df[[
            "property_name", "society", "price", "predicted_future_price",
            "predicted_profit", "roi_percent", "Green Area", "Safety",
            "Connectivity", "avg_rating"
        ]]
        .sort_values(by="roi_percent", ascending=False)
        .head(20)
        .style.format({
            "price": "₹{:,.0f}",
            "predicted_future_price": "₹{:,.0f}",
            "predicted_profit": "₹{:,.0f}",
            "roi_percent": "{:.2f}",
            "avg_rating": "{:.1f}"
        }),
        use_container_width=True
    )
    
    st.markdown("### 🧭 Investment Insights")
    st.write(f"""
    - 📊 *Average ROI:* Around *{df['roi_percent'].mean():.2f}%*
    - 🏡 *Top Society:* {roi_by_society.iloc[0]['society']} with ROI *{roi_by_society.iloc[0]['roi_percent']:.2f}%*
    - 💰 *Max Predicted Profit:* ₹{df['predicted_profit'].max():,.0f}
    - 🌱 *Best Environment Rating Avg:* {df['Green Area'].mean():.2f}
    - 🚆 *Connectivity Score Avg:* {df['Connectivity'].mean():.2f}
    """)
    
    st.success("✅ ROI & Prediction Analysis Completed Successfully!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("<p style='text-align:center;'>© 2025 Real Estate Dashboard | Developed by Anuj Tiwari</p>", unsafe_allow_html=True)