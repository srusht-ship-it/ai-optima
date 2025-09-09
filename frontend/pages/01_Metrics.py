import requests
import pandas as pd
import streamlit as st

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
st.title("📊 Metrics")

@st.cache_data(ttl=5)
def fetch_metrics():
    r = requests.get(f"{API_BASE}/agent-stats", timeout=10)
    r.raise_for_status()
    return r.json()

try:
    m = fetch_metrics()
    c1, c2, c3, c4 = st.columns(4)
    
    # Use .get() with default values to handle missing keys
    c1.metric("Total Classifications", f"{m.get('total_classifications', 0):,}")
    c2.metric("Escalation Rate", f"{m.get('escalation_rate', 0)*100:.1f}%")
    c3.metric("Avg CO₂/Email (g)", f"{m.get('avg_co2_per_email_g', 0):.4f}")
    c4.metric("Total CO₂ (g)", f"{m.get('total_co2_emissions_g', 0):.4f}")

    # Display model performance
    df_models = pd.DataFrame(m.get("model_performance", []))
    if not df_models.empty:
        st.subheader("Model Performance")
        st.dataframe(df_models.sort_values("usage_count", ascending=False), use_container_width=True)
        st.subheader("Usage by Model")
        st.bar_chart(df_models.set_index("model")["usage_count"])

    # Display category distribution
    df_categories = pd.DataFrame(m.get("category_distribution", []))
    if not df_categories.empty:
        st.subheader("Category Distribution")
        st.dataframe(df_categories.sort_values("count", ascending=False), use_container_width=True)
        st.bar_chart(df_categories.set_index("category")["count"])

    if df_models.empty and df_categories.empty:
        st.info("No data yet. Make some /classify-email requests.")
        
except Exception as e:
    st.error(f"Failed to load metrics from {API_BASE}: {e}")
    
    # Optional: Add debug info to see what the API actually returns
    if st.checkbox("Show debug info"):
        try:
            r = requests.get(f"{API_BASE}/agent-stats", timeout=10)
            st.json(r.json())
        except Exception as debug_e:
            st.error(f"Debug request failed: {debug_e}")