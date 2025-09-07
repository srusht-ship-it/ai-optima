import requests, streamlit as st
import json

API_BASE = st.secrets.get("API_BASE", "http://127.0.0.1:8000")
st.title("📧 Email Classifier")

st.write("Intelligent email classification with energy optimization")

# Input section
col1, col2 = st.columns([2, 1])

with col1:
    email_text = st.text_area(
        "Email Content", 
        placeholder="Paste your email content here...",
        height=200
    )
    
    subject = st.text_input("Subject (optional)", placeholder="Email subject")
    sender = st.text_input("Sender (optional)", placeholder="sender@example.com")

with col2:
    st.subheader("Preferences")
    priority = st.selectbox("Priority", ["balanced", "accuracy", "speed", "energy"])
    confidence_threshold = st.slider("Confidence Threshold", 0.5, 0.99, 0.85, 0.01)
    
    preferences = {
        "priority": priority,
        "confidence_threshold": confidence_threshold
    }

if st.button("🔍 Classify Email", use_container_width=True):
    if not email_text.strip():
        st.error("Please enter email content")
    else:
        with st.spinner("Analyzing email..."):
            payload = {
                "text": email_text,
                "subject": subject,
                "sender": sender,
                "preferences": preferences
            }
            
            try:
                response = requests.post(f"{API_BASE}/classify-email", json=payload, timeout=30)
                response.raise_for_status()
                result = response.json()
                
                # Display results
                st.success("✅ Classification Complete!")
                
                # Main result
                col1, col2, col3 = st.columns(3)
                col1.metric("📂 Category", result["predicted_category"].title())
                col2.metric("🎯 Confidence", f"{result['confidence']:.1%}")
                col3.metric("⚡ Model Used", "Heavy" if "bert-base" in result["model_used"] else "Light")
                
                # Energy metrics
                st.subheader("🌱 Energy Impact")
                energy = result["energy_metrics"]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("🔥 CO₂ Emissions", f"{energy['co2_emissions_g']:.3f}g")
                col2.metric("⏱ Processing Time", f"{energy['processing_time_seconds']:.2f}s")
                col3.metric("📊 Efficiency Score", f"{energy['energy_efficiency_score']:.2f}")
                col4.metric("🔄 Escalated", "Yes" if result["escalated"] else "No")
                
                # AI Insights
                insights = result["ai_insights"]
                
                st.subheader("🧠 AI Insights")
                
                # Environmental impact
                env_impact = insights["environmental_impact"]
                st.info(f"🌍 *Environmental Impact*: {env_impact['impact_level'].title()} "
                       f"({env_impact['co2_this_classification']:.3f}g CO₂)")
                
                if env_impact['yearly_projection_g'] > 100:
                    st.warning(f"📈 Yearly projection: {env_impact['yearly_projection_g']:.1f}g CO₂ "
                             f"(≈ {env_impact['equivalent_km_driven']:.1f}km of car driving)")
                
                # Suggestions
                if insights["suggestions"]:
                    st.subheader("💡 Suggestions")
                    for suggestion in insights["suggestions"]:
                        st.write(f"• {suggestion}")
                
                # All predictions
                st.subheader("📊 All Predictions")
                predictions_df = pd.DataFrame(result["all_predictions"])
                st.bar_chart(predictions_df.set_index("category")["confidence"])
                
                # Raw response (expandable)
                with st.expander("🔧 Technical Details"):
                    st.json(result)
                    
            except Exception as e:
                st.error(f"❌ Classification failed: {e}")

# Performance monitoring
st.subheader("📈 Quick Stats")
try:
    stats_response = requests.get(f"{API_BASE}/agent-stats", timeout=10)
    if stats_response.status_code == 200:
        stats = stats_response.json()
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Classifications", f"{stats['total_classifications']:,}")
        col2.metric("Escalation Rate", f"{stats['escalation_rate']:.1%}")
        col3.metric("Avg CO₂/Email", f"{stats['avg_co2_per_email_g']:.3f}g")
        col4.metric("Energy Saved", f"{stats['energy_savings_estimate']:.1f}g")
        
except:
    st.info("Stats unavailable - start classifying emails to see performance metrics!")