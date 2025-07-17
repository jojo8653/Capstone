#!/usr/bin/env python3
"""
HotelOptix Disaster Response Tool - Command Line Demo
Demonstrates emergency rebooking and service worker relocation capabilities
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare hotel booking data"""
    print("📊 Loading hotel booking data...")
    
    df = pd.read_csv('hotel_bookings.csv')
    
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
    
    print(f"✅ Loaded {len(df):,} bookings with {df['disaster_affected'].sum():,} disaster-affected")
    return df

def train_random_forest_model(df):
    """Train Random Forest model for cancellation prediction"""
    print("🤖 Training Random Forest model...")
    
    # Encode categorical variables
    categorical_cols = ['hotel', 'meal', 'country', 'market_segment', 'distribution_channel',
                       'reserved_room_type', 'deposit_type', 'customer_type']
    
    df_encoded = df.copy()
    for col in categorical_cols:
        if col in df_encoded.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
    
    # Features for modeling
    feature_cols = [
        'hotel', 'lead_time', 'arrival_month', 'total_nights', 'total_guests',
        'meal', 'country', 'market_segment', 'distribution_channel',
        'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled',
        'reserved_room_type', 'booking_changes', 'deposit_type', 'customer_type',
        'required_car_parking_spaces', 'total_of_special_requests',
        'emergency_booking', 'emergency_priority'
    ]
    
    available_features = [col for col in feature_cols if col in df_encoded.columns]
    X = df_encoded[available_features]
    y = df_encoded['is_canceled']
    
    # Train model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = accuracy_score(y_test, model.predict(X_test))
    print(f"✅ Random Forest trained with {accuracy:.3f} accuracy")
    
    return model, available_features

def find_emergency_alternatives(df, disaster_location, guest_requirements):
    """Find alternative hotels for emergency rebooking"""
    print(f"\n🚨 Finding emergency alternatives for guests from {disaster_location}...")
    
    # Filter safe locations
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
        'total_guests': 'count'
    }).reset_index()
    
    hotel_availability['availability_score'] = (
        hotel_availability['is_canceled'] * 0.6 +
        (1 - hotel_availability['total_guests'] / hotel_availability['total_guests'].max()) * 0.4
    )
    
    # Emergency pricing
    emergency_hotels = hotel_availability[hotel_availability['availability_score'] > 0.3].copy()
    emergency_hotels['emergency_price'] = emergency_hotels['adr'] * 1.2  # 20% surge
    
    print(f"✅ Found {len(emergency_hotels)} emergency accommodation options")
    return emergency_hotels.sort_values('availability_score', ascending=False)

def relocate_service_workers(df, disaster_location, num_workers, duration_days):
    """Find accommodation for displaced service workers"""
    print(f"\n👷 Finding accommodation for {num_workers} service workers for {duration_days} days...")
    
    # Find budget-friendly options in safe areas
    safe_areas = df[df['country'] != disaster_location].copy()
    worker_suitable = safe_areas[
        (safe_areas['adr'] <= safe_areas['adr'].quantile(0.4)) &  # Budget-friendly
        (safe_areas['total_nights'] >= 7)  # Longer stays
    ]
    
    # Group by location and calculate suitability
    worker_accommodations = worker_suitable.groupby(['country', 'hotel']).agg({
        'adr': 'mean',
        'is_canceled': 'mean',
        'total_guests': 'count'
    }).reset_index()
    
    worker_accommodations['worker_suitability'] = (
        (worker_accommodations['adr'] <= worker_accommodations['adr'].quantile(0.5)) * 0.4 +
        worker_accommodations['is_canceled'] * 0.3 +
        (worker_accommodations['total_guests'] / worker_accommodations['total_guests'].max()) * 0.3
    )
    
    worker_accommodations['total_cost'] = worker_accommodations['adr'] * duration_days * num_workers
    
    print(f"✅ Found {len(worker_accommodations)} suitable worker accommodation options")
    return worker_accommodations.sort_values('worker_suitability', ascending=False)

def analyze_disaster_impact(df, disaster_location):
    """Analyze the impact of the disaster"""
    print(f"\n📊 Analyzing disaster impact in {disaster_location}...")
    
    affected_bookings = df[df['disaster_affected'] == 1]
    location_affected = affected_bookings[affected_bookings['country'] == disaster_location]
    
    print(f"📈 DISASTER IMPACT ANALYSIS")
    print(f"   Total affected bookings: {len(affected_bookings):,}")
    print(f"   Affected in {disaster_location}: {len(location_affected):,}")
    print(f"   High priority evacuations: {len(affected_bookings[affected_bookings['emergency_priority'] >= 3]):,}")
    print(f"   Total guests affected: {affected_bookings['total_guests'].sum():,}")
    print(f"   Average booking value: ${affected_bookings['adr'].mean():.2f}")
    print(f"   Total revenue at risk: ${affected_bookings['adr'].sum():,.2f}")
    
    # Priority breakdown
    priority_dist = affected_bookings['emergency_priority'].value_counts().sort_index()
    print(f"\n🚨 EVACUATION PRIORITY BREAKDOWN:")
    for priority, count in priority_dist.items():
        priority_level = ["Low", "Medium", "High", "Critical"][min(int(priority), 3)]
        print(f"   Priority {priority} ({priority_level}): {count:,} bookings")

def main():
    """Main demo function"""
    print("🏨 HotelOptix Disaster Response Tool - Demo")
    print("=" * 50)
    
    # Load data and train model
    df = load_and_prepare_data()
    model, features = train_random_forest_model(df)
    
    # Simulate disaster scenario
    disaster_location = 'PRT'  # Portugal
    disaster_type = 'Hurricane'
    
    print(f"\n🌪️  DISASTER SCENARIO: {disaster_type} affecting {disaster_location}")
    print("=" * 50)
    
    # Analyze impact
    analyze_disaster_impact(df, disaster_location)
    
    # Emergency rebooking demo
    guest_requirements = {
        'total_guests': 4,  # Family of 4
        'total_nights': 5,  # 5-night stay
        'budget_max': 150   # Max $150/night
    }
    
    emergency_alternatives = find_emergency_alternatives(df, disaster_location, guest_requirements)
    
    print(f"\n🏨 TOP 5 EMERGENCY REBOOKING OPTIONS:")
    print("-" * 50)
    for i, (_, hotel) in enumerate(emergency_alternatives.head(5).iterrows(), 1):
        print(f"{i}. {hotel['hotel']} in {hotel['country']}")
        print(f"   💰 Emergency Rate: ${hotel['emergency_price']:.0f}/night")
        print(f"   📊 Availability Score: {hotel['availability_score']:.2f}")
        print(f"   📈 Historical Cancellations: {hotel['is_canceled']*100:.1f}%")
        print()
    
    # Service worker relocation demo
    num_workers = 15
    duration_days = 30
    
    worker_accommodations = relocate_service_workers(df, disaster_location, num_workers, duration_days)
    
    print(f"\n👷 TOP 5 SERVICE WORKER ACCOMMODATION OPTIONS:")
    print("-" * 50)
    for i, (_, accommodation) in enumerate(worker_accommodations.head(5).iterrows(), 1):
        print(f"{i}. {accommodation['hotel']} in {accommodation['country']}")
        print(f"   💰 Rate: ${accommodation['adr']:.0f}/night per worker")
        print(f"   💵 Total Cost: ${accommodation['total_cost']:,.0f} for {num_workers} workers")
        print(f"   ⭐ Suitability Score: {accommodation['worker_suitability']:.2f}")
        print()
    
    # Model insights
    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n🤖 RANDOM FOREST MODEL INSIGHTS:")
    print("-" * 50)
    print("Top 5 Most Important Features:")
    for i, (_, feat) in enumerate(feature_importance.head(5).iterrows(), 1):
        print(f"{i}. {feat['feature']}: {feat['importance']:.3f}")
    
    print(f"\n✅ DEMO COMPLETE")
    print("=" * 50)
    print("💡 To launch the interactive Streamlit dashboard:")
    print("   streamlit run disaster_response_tool.py")
    print("\n📋 This tool provides:")
    print("   • Real-time emergency rebooking")
    print("   • Service worker relocation coordination") 
    print("   • Interactive analytics dashboard")
    print("   • AI-powered risk assessment")

if __name__ == "__main__":
    main() 