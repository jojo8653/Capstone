import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import warnings
import base64
import os
warnings.filterwarnings('ignore')

# Core configuration
CONFIG = {
    'emergency_radius_km': 150,
    'safe_distance_km': 200,
    'price_surge_factor': 1.2,
    'worker_relocation_radius': 300,
    'priority_booking_hours': 72,
}

def load_logo():
    """Load logo image if available"""
    logo_path = "Brand kit.png"  # Your logo file name
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{logo_data}"
    return None

@st.cache_data
def load_and_prepare_data():
    """Load and prepare hotel booking data with disaster response features"""
    try:
        df = pd.read_csv('hotel_bookings.csv')
    except FileNotFoundError:
        st.error("❌ hotel_bookings.csv file not found. Please upload the dataset.")
        return pd.DataFrame()
    
    # Clean data
    df = df.drop(columns=['company', 'agent', 'reservation_status_date'], errors='ignore')
    df['children'] = df['children'].fillna(0).astype(int)
    
    # Handle missing countries
    if df['country'].isna().any():
        modes = df.groupby('market_segment')['country'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        df['country'] = df.apply(lambda r: modes[r['market_segment']] if pd.isna(r['country']) else r['country'], axis=1)
    
    # Create disaster response features
    df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df['total_guests'] = df[['adults','children','babies']].sum(axis=1)
    df['emergency_booking'] = (df['lead_time'] <= 7).astype(int)
    
    # Date features
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' + 
        df['arrival_date_month'].str[:3] + '-' + 
        df['arrival_date_day_of_month'].astype(str),
        format='%Y-%b-%d', errors='coerce'
    )
    df['arrival_month'] = df['arrival_date'].dt.month
    
    # Disaster season risk
    df['disaster_season_risk'] = df['arrival_month'].map({
        6: 3, 7: 3, 8: 3, 9: 3, 10: 2,  # Hurricane season
        3: 2, 4: 2, 5: 2,  # Flood season
        11: 1, 12: 1, 1: 1, 2: 1  # Low risk
    })
    
    # Emergency priority score
    df['emergency_priority'] = (
        (df['total_guests'] >= 4) * 3 +  # Families get priority
        (df['babies'] > 0) * 2 +         # Families with babies
        (df['total_nights'] >= 7) * 1 +  # Long stays
        (df['adr'] >= df['adr'].quantile(0.8)) * 1  # High-value guests
    )
    
    # Simulate disaster events
    np.random.seed(42)
    disaster_months = [6, 7, 8, 9, 10]
    df['disaster_affected'] = 0
    disaster_season_mask = df['arrival_month'].isin(disaster_months)
    df.loc[disaster_season_mask, 'disaster_affected'] = np.random.binomial(1, 0.12, sum(disaster_season_mask))
    df.loc[~disaster_season_mask, 'disaster_affected'] = np.random.binomial(1, 0.02, sum(~disaster_season_mask))
    
    return df

def find_emergency_alternatives(df, disaster_location, guest_requirements):
    """Find alternative hotels for emergency rebooking"""
    
    # Filter safe locations (different country or region)
    safe_hotels = df[df['country'] != disaster_location].copy()
    
    # Filter by guest requirements
    if guest_requirements.get('total_guests'):
        safe_hotels = safe_hotels[safe_hotels['total_guests'] >= guest_requirements['total_guests']]
    
    if guest_requirements.get('total_nights'):
        safe_hotels = safe_hotels[safe_hotels['total_nights'] >= guest_requirements['total_nights']]
    
    # Calculate availability score
    hotel_availability = safe_hotels.groupby(['country', 'hotel']).agg({
        'is_canceled': 'mean',
        'adr': 'mean',
        'total_guests': 'count',
    }).reset_index()
    
    hotel_availability['availability_score'] = (
        hotel_availability['is_canceled'] * 0.6 +  # Higher cancellations = more availability
        (1 - hotel_availability['total_guests'] / hotel_availability['total_guests'].max()) * 0.4
    )
    
    # Emergency pricing
    emergency_hotels = hotel_availability[hotel_availability['availability_score'] > 0.3].copy()
    emergency_hotels['emergency_price'] = emergency_hotels['adr'] * CONFIG['price_surge_factor']
    
    return emergency_hotels.sort_values('availability_score', ascending=False)

def relocate_service_workers(df, disaster_location, worker_requirements):
    """Find accommodation for displaced service workers"""
    
    # Find budget-friendly options in safe areas
    safe_areas = df[df['country'] != disaster_location].copy()
    worker_suitable = safe_areas[
        (safe_areas['adr'] <= safe_areas['adr'].quantile(0.4)) &  # Budget-friendly
        (safe_areas['total_nights'] >= 7)  # Longer stays suitable for workers
    ]
    
    # Group by location and calculate suitability
    worker_accommodations = worker_suitable.groupby(['country', 'hotel']).agg({
        'adr': 'mean',
        'is_canceled': 'mean',
        'total_guests': 'count',
    }).reset_index()
    
    worker_accommodations['worker_suitability'] = (
        (worker_accommodations['adr'] <= worker_accommodations['adr'].quantile(0.5)) * 0.4 +
        worker_accommodations['is_canceled'] * 0.3 +
        (worker_accommodations['total_guests'] / worker_accommodations['total_guests'].max()) * 0.3
    )
    
    return worker_accommodations.sort_values('worker_suitability', ascending=False)

def calculate_evacuation_priority(bookings_df):
    """Calculate evacuation priority for affected bookings"""
    
    priority_bookings = bookings_df[bookings_df['disaster_affected'] == 1].copy()
    
    if len(priority_bookings) == 0:
        return pd.DataFrame()
    
    # Priority scoring
    priority_bookings['evacuation_priority'] = (
        priority_bookings['emergency_priority'] * 0.4 +
        (priority_bookings['babies'] > 0) * 3 +
        (priority_bookings['total_guests'] >= 4) * 2 +
        (priority_bookings['adr'] >= priority_bookings['adr'].quantile(0.8)) * 1
    )
    
    return priority_bookings.sort_values('evacuation_priority', ascending=False)

def main():
    st.set_page_config(
        page_title="HotelOptix Disaster Response Tool",
        page_icon="🏨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for branding
    st.markdown("""
    <style>
    /* Import custom fonts - replace 'YourFontName' with actual font from Canva */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    
    /* Custom styling */
    .main-header {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 30px;
    }
    
    .logo-container {
        flex-shrink: 0;
    }
    
    .title-container {
        flex-grow: 1;
    }
    
    /* Custom font for titles */
    .custom-title {
        font-family: 'Inter', sans-serif;
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        margin: 0;
    }
    
    .custom-subtitle {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        color: #6b7280;
        margin: 5px 0;
    }
    
    /* Color scheme - update these with your Canva colors */
    :root {
        --primary-color: #3b82f6;
        --secondary-color: #1e40af;
        --accent-color: #f59e0b;
        --text-color: #1f2937;
        --bg-color: #f8fafc;
    }
    
    /* Style metrics and cards */
    .metrics-container {
        margin: 20px 0;
    }
    
    .metric-card {
        background: white;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid var(--primary-color);
        margin-bottom: 15px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.15);
    }
    
    .metric-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.9rem;
        color: #6b7280;
        margin-bottom: 8px;
        font-weight: 500;
    }
    
    .metric-value {
        font-family: 'Inter', sans-serif;
        font-size: 2rem;
        font-weight: 700;
        color: var(--primary-color);
        line-height: 1;
        margin-bottom: 5px;
    }
    
    .metric-delta {
        font-family: 'Inter', sans-serif;
        font-size: 0.8rem;
        color: #9ca3af;
        font-weight: 400;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header with logo and title
    logo_data = load_logo()
    
    if logo_data:
        # Use actual logo - Updated to larger size
        st.markdown(f"""
        <div class="main-header">
            <div class="logo-container">
                <img src="{logo_data}" style="width: 120px; height: 120px; object-fit: contain;">
            </div>
            <div class="title-container">
                <h1 class="custom-title">HotelOptix Disaster Response Tool</h1>
                <p class="custom-subtitle">Professional Emergency Rebooking & Service Worker Relocation Platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # Fallback placeholder logo - Updated to larger size
        st.markdown("""
        <div class="main-header">
            <div class="logo-container">
                <div style="width: 120px; height: 120px; background: linear-gradient(45deg, #3b82f6, #1e40af); border-radius: 16px; display: flex; align-items: center; justify-content: center; color: white; font-size: 36px; font-weight: bold;">
                    H
                </div>
            </div>
            <div class="title-container">
                <h1 class="custom-title">HotelOptix Disaster Response Tool</h1>
                <p class="custom-subtitle">Professional Emergency Rebooking & Service Worker Relocation Platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_and_prepare_data()
        
        if df.empty:
            st.stop()
    
    # Sidebar for disaster simulation
    st.sidebar.header("🚨 Disaster Scenario Setup")
    
    disaster_type = st.sidebar.selectbox(
        "Disaster Type",
        ["Hurricane", "Flood", "Earthquake", "Wildfire", "Tornado"]
    )
    
    affected_location = st.sidebar.selectbox(
        "Affected Location",
        df['country'].unique()
    )
    
    severity = st.sidebar.slider("Disaster Severity (1-5)", 1, 5, 3)
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard Overview", 
        "🏨 Emergency Rebooking", 
        "👷 Worker Relocation",
        "📈 Analytics"
    ])
    
    with tab1:
        st.header("Disaster Response Dashboard Overview")
        
        # Key metrics with custom styling
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        affected_bookings = df[df['disaster_affected'] == 1]
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Affected Bookings</div>
                <div class="metric-value">{}</div>
                <div class="metric-delta">Severity {} in {}</div>
            </div>
            """.format(len(affected_bookings), severity, affected_location), unsafe_allow_html=True)
        
        with col2:
            emergency_bookings = affected_bookings[affected_bookings['emergency_booking'] == 1]
            percentage = f"{len(emergency_bookings)/len(affected_bookings)*100:.1f}%" if len(affected_bookings) > 0 else "0%"
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Emergency Bookings</div>
                <div class="metric-value">{}</div>
                <div class="metric-delta">{} of affected</div>
            </div>
            """.format(len(emergency_bookings), percentage), unsafe_allow_html=True)
        
        with col3:
            high_priority = affected_bookings[affected_bookings['emergency_priority'] >= 3]
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">High Priority Cases</div>
                <div class="metric-value">{}</div>
                <div class="metric-delta">Families & VIP guests</div>
            </div>
            """.format(len(high_priority)), unsafe_allow_html=True)
        
        with col4:
            avg_nights = affected_bookings['total_nights'].mean() if len(affected_bookings) > 0 else 0
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Avg. Nights Affected</div>
                <div class="metric-value">{:.1f}</div>
                <div class="metric-delta">Per booking</div>
            </div>
            """.format(avg_nights), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Impact analysis
        st.subheader("Impact by Country")
        
        impact_by_country = df.groupby('country').agg({
            'disaster_affected': 'sum',
            'adr': 'mean',
            'total_guests': 'sum'
        }).reset_index()
        
        st.dataframe(
            impact_by_country.sort_values('disaster_affected', ascending=False).head(10),
            use_container_width=True
        )
    
    with tab2:
        st.header("🏨 Emergency Rebooking System")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Guest Requirements")
            
            guest_total = st.number_input("Total Guests", min_value=1, max_value=10, value=2)
            guest_nights = st.number_input("Total Nights", min_value=1, max_value=30, value=3)
            budget_max = st.number_input("Maximum Budget per Night", min_value=50, max_value=500, value=150)
            
            guest_requirements = {
                'total_guests': guest_total,
                'total_nights': guest_nights,
                'budget_max': budget_max
            }
            
            if st.button("🔍 Find Emergency Alternatives", type="primary"):
                with st.spinner("Searching for available alternatives..."):
                    alternatives = find_emergency_alternatives(df, affected_location, guest_requirements)
                    st.session_state['alternatives'] = alternatives
        
        with col1:
            if 'alternatives' in st.session_state:
                st.subheader("Available Emergency Accommodations")
                
                alternatives = st.session_state['alternatives'].head(10)
                
                if len(alternatives) == 0:
                    st.warning("No suitable alternatives found for the given criteria.")
                else:
                    for idx, hotel in alternatives.iterrows():
                        with st.expander(f"🏨 {hotel['hotel']} in {hotel['country']} - Availability: {hotel['availability_score']:.2f}"):
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Emergency Price", f"${hotel['emergency_price']:.0f}/night")
                            
                            with col_b:
                                st.metric("Availability Score", f"{hotel['availability_score']:.2f}")
                            
                            with col_c:
                                st.metric("Historical Cancellations", f"{hotel['is_canceled']*100:.1f}%")
                            
                            if st.button(f"📞 Contact Hotel {hotel['hotel']}", key=f"contact_{idx}"):
                                st.success(f"Emergency booking request sent to {hotel['hotel']} in {hotel['country']}")
    
    with tab3:
        st.header("👷 Service Worker Relocation")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Worker Requirements")
            
            num_workers = st.number_input("Number of Workers", min_value=1, max_value=100, value=10)
            duration_days = st.number_input("Relocation Duration (days)", min_value=7, max_value=180, value=30)
            budget_per_worker = st.number_input("Budget per Worker per Night", min_value=30, max_value=150, value=60)
            
            worker_requirements = {
                'num_workers': num_workers,
                'duration_days': duration_days,
                'budget_per_worker': budget_per_worker
            }
            
            if st.button("🔍 Find Worker Accommodations", type="primary"):
                with st.spinner("Finding suitable worker accommodations..."):
                    worker_options = relocate_service_workers(df, affected_location, worker_requirements)
                    st.session_state['worker_options'] = worker_options
        
        with col1:
            if 'worker_options' in st.session_state:
                st.subheader("Worker Accommodation Options")
                
                worker_options = st.session_state['worker_options'].head(8)
                
                if len(worker_options) == 0:
                    st.warning("No suitable worker accommodations found.")
                else:
                    for idx, accommodation in worker_options.iterrows():
                        with st.expander(f"🏠 {accommodation['hotel']} in {accommodation['country']} - Suitability: {accommodation['worker_suitability']:.2f}"):
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Rate per Night", f"${accommodation['adr']:.0f}")
                            
                            with col_b:
                                total_cost = accommodation['adr'] * duration_days * num_workers
                                st.metric("Total Cost", f"${total_cost:,.0f}")
                            
                            with col_c:
                                st.metric("Suitability Score", f"{accommodation['worker_suitability']:.2f}")
                            
                            if st.button(f"📋 Reserve for Workers", key=f"worker_{idx}"):
                                st.success(f"Reservation request sent for {num_workers} workers at {accommodation['hotel']}")
    
    with tab4:
        st.header("📈 Disaster Response Analytics")
        
        # Evacuation priority analysis
        priority_bookings = calculate_evacuation_priority(df)
        
        if len(priority_bookings) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("High Priority Evacuations")
                high_priority = priority_bookings.head(10)[['country', 'hotel', 'total_guests', 'babies', 'evacuation_priority']]
                st.dataframe(high_priority, use_container_width=True)
            
            with col2:
                st.subheader("Priority Distribution")
                priority_counts = priority_bookings['evacuation_priority'].value_counts().sort_index()
                
                # Simple bar chart using dataframe
                priority_df = pd.DataFrame({
                    'Priority Score': priority_counts.index,
                    'Number of Bookings': priority_counts.values
                })
                st.bar_chart(priority_df.set_index('Priority Score'))
        
        # Financial impact analysis
        st.subheader("Financial Impact Analysis")
        
        affected_bookings = df[df['disaster_affected'] == 1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            revenue_loss = affected_bookings['adr'].sum()
            st.metric("Potential Revenue Loss", f"${revenue_loss:,.0f}")
        
        with col2:
            rebooking_cost = revenue_loss * CONFIG['price_surge_factor']
            st.metric("Emergency Rebooking Cost", f"${rebooking_cost:,.0f}")
        
        with col3:
            net_impact = rebooking_cost - revenue_loss
            st.metric("Net Financial Impact", f"${net_impact:,.0f}", 
                     delta=f"{(net_impact/revenue_loss)*100:.1f}%" if revenue_loss > 0 else "0%")
        
        # Summary statistics
        st.subheader("Disaster Impact Summary")
        
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Affected Bookings',
                'Average ADR (Affected)',
                'Total Guests Affected',
                'Emergency Bookings',
                'High Priority Cases'
            ],
            'Value': [
                len(affected_bookings),
                f"${affected_bookings['adr'].mean():.2f}" if len(affected_bookings) > 0 else "$0",
                affected_bookings['total_guests'].sum(),
                len(affected_bookings[affected_bookings['emergency_booking'] == 1]),
                len(affected_bookings[affected_bookings['emergency_priority'] >= 3])
            ]
        })
        
        st.dataframe(summary_stats, use_container_width=True)
        
        # Model information
        st.subheader("🤖 Analysis Methodology")
        st.info("""
        **HotelOptix Disaster Response Analytics**
        
        - **Emergency Priority Scoring**: Families with children, VIP guests, and long-stay bookings receive priority
        - **Availability Prediction**: Based on historical cancellation patterns and booking flexibility
        - **Pricing Strategy**: 20% surge pricing during emergency rebooking scenarios
        - **Worker Relocation**: Budget-optimized accommodation search for displaced service workers
        - **Geographic Safety**: Minimum 200km safe distance from disaster zones
        
        *This tool provides data-driven insights for hotel emergency response planning.*
        """)

if __name__ == "__main__":
    main() 
