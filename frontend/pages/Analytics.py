import streamlit as st
import requests
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
st.title("📊 Advanced Analytics")

# Sidebar filters
st.sidebar.header("🔍 Filters")
time_range = st.sidebar.selectbox("Time Range", 
    ["Last 24 hours", "Last 7 days", "Last 30 days", "All time"])
category_filter = st.sidebar.multiselect("Categories", 
    ["work", "spam", "promotions", "personal", "support", "newsletter"])

# Main dashboard
try:
    # Get comprehensive stats
    stats_response = requests.get(f"{API_BASE}/agent-stats", timeout=10)
    learning_response = requests.get(f"{API_BASE}/learning-insights", timeout=10)
    
    if stats_response.status_code == 200:
        stats = stats_response.json()
        
        # Key metrics row
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("📧 Total Emails", f"{stats['total_classifications']:,}")
        
        with col2:
            escalation_rate = stats['escalation_rate']
            st.metric("🔄 Escalation Rate", f"{escalation_rate:.1%}")
        
        with col3:
            avg_co2 = stats['avg_co2_per_email_g']
            st.metric("🌱 Avg CO₂/Email", f"{avg_co2:.3f}g")
        
        with col4:
            total_co2 = stats['total_co2_emissions_g']
            st.metric("🏭 Total CO₂", f"{total_co2:.2f}g")
        
        with col5:
            energy_saved = stats.get('energy_savings_estimate', 0)
            st.metric("💚 Energy Saved", f"{energy_saved:.1f}g")
        
        # Charts section
        st.markdown("---")
        
        # Model performance comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🤖 Model Performance")
            
            if stats.get('model_performance'):
                model_df = pd.DataFrame(stats['model_performance'])
                
                fig = make_subplots(specs=[[{"secondary_y": True}]])
                
                fig.add_trace(
                    go.Bar(name="Usage Count", x=model_df['model'], y=model_df['count']),
                    secondary_y=False,
                )
                
                fig.add_trace(
                    go.Scatter(name="Avg CO₂ (g)", x=model_df['model'], y=model_df['avg_co2_g'], mode='markers+lines'),
                    secondary_y=True,
                )
                
                fig.update_xaxes(title_text="Model")
                fig.update_yaxes(title_text="Usage Count", secondary_y=False)
                fig.update_yaxes(title_text="CO₂ Emissions (g)", secondary_y=True)
                
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📂 Category Distribution")
            
            if stats.get('category_distribution'):
                cat_df = pd.DataFrame(stats['category_distribution'])
                
                fig = px.pie(cat_df, values='count', names='category', 
                           title="Email Categories")
                st.plotly_chart(fig, use_container_width=True)
        
        # Learning insights
        if learning_response.status_code == 200:
            learning_data = learning_response.json()
            
            st.markdown("---")
            st.subheader("🧠 Continuous Learning Insights")
            
            perf_metrics = learning_data.get('performance_metrics', {})
            
            if not perf_metrics.get('insufficient_data'):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    overall_acc = perf_metrics.get('overall_accuracy', 0)
                    st.metric("🎯 Overall Accuracy", f"{overall_acc:.1%}")
                
                with col2:
                    light_acc = perf_metrics.get('light_model_accuracy', 0)
                    st.metric("⚡ Light Model Accuracy", f"{light_acc:.1%}")
                
                with col3:
                    heavy_acc = perf_metrics.get('heavy_model_accuracy', 0)
                    st.metric("🔥 Heavy Model Accuracy", f"{heavy_acc:.1%}")
                
                # Improvement suggestions
                suggestions = perf_metrics.get('improvement_suggestions', [])
                if suggestions:
                    st.subheader("💡 Improvement Suggestions")
                    for suggestion in suggestions:
                        st.info(f"• {suggestion}")
                
                # Threshold recommendations
                threshold_rec = learning_data.get('threshold_recommendation', {})
                if not threshold_rec.get('insufficient_data'):
                    st.subheader("⚙ Threshold Optimization")
                    
                    current_threshold = threshold_rec.get('current_threshold', 0.85)
                    recommended_threshold = threshold_rec.get('recommended_threshold', 0.85)
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Current Threshold", f"{current_threshold:.2f}")
                    col2.metric("Recommended Threshold", f"{recommended_threshold:.2f}")
                    
                    if abs(recommended_threshold - current_threshold) > 0.02:
                        if recommended_threshold > current_threshold:
                            st.warning(f"💡 Consider increasing threshold to {recommended_threshold:.2f} for better precision")
                        else:
                            st.info(f"💡 Consider decreasing threshold to {recommended_threshold:.2f} for better energy efficiency")
            else:
                st.info("🔄 Not enough feedback data for learning insights. Start providing feedback on classifications!")
        
        # Environmental Impact Analysis
        st.markdown("---")
        st.subheader("🌍 Environmental Impact Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # CO₂ impact comparison
            daily_emails = 50  # Average
            yearly_co2 = avg_co2 * daily_emails * 365
            
            # Comparison with other activities
            comparison_data = {
                "Activity": ["Email Classification (Yearly)", "1km Car Drive", "1 Google Search", "1 Email Send"],
                "CO₂ (g)": [yearly_co2, 404, 0.2, 4]
            }
            
            comp_df = pd.DataFrame(comparison_data)
            fig = px.bar(comp_df, x="Activity", y="CO₂ (g)", 
                        title="CO₂ Emissions Comparison")
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Energy efficiency over time (simulated trend)
            dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
            efficiency_scores = np.random.normal(1.2, 0.3, 30)  # Simulated data
            efficiency_scores = np.maximum(0.5, efficiency_scores)  # Keep positive
            
            trend_df = pd.DataFrame({
                "Date": dates,
                "Efficiency Score": efficiency_scores
            })
            
            fig = px.line(trend_df, x="Date", y="Efficiency Score", 
                         title="Energy Efficiency Trend (30 days)")
            fig.add_hline(y=1.0, line_dash="dash", line_color="green", 
                         annotation_text="Target Efficiency")
            st.plotly_chart(fig, use_container_width=True)
        
        # Real-time monitoring section
        st.markdown("---")
        st.subheader("📡 Real-time Monitoring")
        
        # Auto-refresh option
        auto_refresh = st.checkbox("Auto-refresh every 30 seconds")
        if auto_refresh:
            st.rerun()
        
        # Recent activity
        try:
            recent_response = requests.get(f"{API_BASE}/runs/recent?limit=10", timeout=5)
            if recent_response.status_code == 200:
                recent_data = recent_response.json()
                if recent_data:
                    recent_df = pd.DataFrame(recent_data)
                    
                    # Show recent classifications
                    st.subheader("🕒 Recent Classifications")
                    
                    display_cols = ['ts_local', 'predicted_category', 'confidence', 'model_used', 'escalated', 'co2_g']
                    if all(col in recent_df.columns for col in display_cols):
                        display_df = recent_df[display_cols].head(5)
                        display_df.columns = ['Time', 'Category', 'Confidence', 'Model', 'Escalated', 'CO₂ (g)']
                        
                        st.dataframe(display_df, use_container_width=True)
                        
                        # Real-time metrics
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            recent_avg_co2 = recent_df['co2_g'].mean()
                            st.metric("Recent Avg CO₂", f"{recent_avg_co2:.3f}g")
                        
                        with col2:
                            recent_escalations = (recent_df['escalated'] == True).sum()
                            st.metric("Recent Escalations", f"{recent_escalations}/10")
                        
                        with col3:
                            recent_avg_conf = recent_df['confidence'].mean()
                            st.metric("Recent Avg Confidence", f"{recent_avg_conf:.1%}")
        except:
            st.info("Recent activity data unavailable")
    
    else:
        st.error("Failed to load analytics data")

except Exception as e:
    st.error(f"Failed to connect to API: {e}")
    st.info("Make sure the backend is running on http://127.0.0.1:8000")

# Export functionality
st.markdown("---")
st.subheader("📤 Export Data")

col1, col2 = st.columns(2)

with col1:
    if st.button("📊 Export Analytics Report"):
        # This would generate a comprehensive report
        st.success("Report generation feature coming soon!")

with col2:
    if st.button("💾 Download Raw Data"):
        # This would allow downloading the classification data
        st.success("Data download feature coming soon!")