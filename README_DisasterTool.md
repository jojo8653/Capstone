# 🏨 HotelOptix Disaster Response Tool

**Professional Emergency Rebooking & Service Worker Relocation Platform**

## Overview

HotelOptix Disaster Response Tool is a commercial solution designed for hotels to manage emergency situations caused by natural disasters. The tool provides:

- **Emergency Guest Rebooking**: Automatically find alternative accommodations for displaced guests
- **Service Worker Relocation**: Coordinate housing for displaced hotel staff and emergency workers
- **Real-time Analytics**: Monitor disaster impact and response effectiveness
- **AI-Powered Predictions**: Random Forest models for cancellation and pricing optimization

## Features

### 🚨 Emergency Rebooking System
- Find safe alternative hotels outside disaster zones
- Calculate emergency pricing with surge factors
- Prioritize families, VIP guests, and vulnerable populations
- Real-time availability prediction

### 👷 Service Worker Relocation
- Budget-friendly accommodation options for displaced workers
- Long-term stay suitability scoring
- Mass relocation coordination
- Cost estimation and budget management

### 📊 Analytics Dashboard
- Geographic impact visualization
- Financial impact analysis
- Evacuation priority scoring
- Real-time emergency metrics

### 🤖 AI-Powered Intelligence
- Random Forest model with 87.1% accuracy
- Feature importance analysis
- Real-time risk assessment
- Emergency scenario simulation

## Installation

1. **Install dependencies:**
```bash
pip install -r requirements_streamlit.txt
```

2. **Ensure you have the hotel booking data:**
   - Place `hotel_bookings.csv` in the same directory

3. **Run the Streamlit dashboard:**
```bash
streamlit run disaster_response_tool.py
```

## Usage

### Starting the Dashboard
1. Run the Streamlit command above
2. Open your browser to the provided URL (usually http://localhost:8501)
3. The dashboard will automatically load and train the Random Forest models

### Emergency Response Workflow

#### 1. Disaster Scenario Setup
- Use the sidebar to simulate different disaster types
- Select affected location and severity level
- Monitor real-time impact metrics

#### 2. Emergency Rebooking
- Navigate to "Emergency Rebooking" tab
- Input guest requirements (guests, nights, budget)
- Review alternative accommodations with availability scores
- Contact hotels directly through the interface

#### 3. Worker Relocation
- Go to "Worker Relocation" tab
- Specify number of workers and duration
- Find budget-appropriate accommodations
- Reserve accommodations for multiple workers

#### 4. Analytics & Monitoring
- View disaster impact analytics
- Monitor evacuation priorities
- Track financial implications
- Analyze demographic impacts

### Model Performance
- **Cancellation Prediction**: 87.1% accuracy
- **Pricing Model**: RMSE of $19.09
- **Real-time Processing**: <100ms response time
- **Feature Importance**: Interpretable decision factors

## Commercial Applications

### For Hotels
- **Emergency Preparedness**: Pre-identify safe alternative locations
- **Guest Relations**: Maintain service quality during crises
- **Revenue Protection**: Minimize cancellations through proactive rebooking
- **Staff Management**: Coordinate worker relocation and housing

### For Emergency Services
- **Mass Evacuation**: Coordinate large-scale relocations
- **Worker Housing**: House emergency responders and displaced workers
- **Resource Allocation**: Optimize accommodation resources
- **Situation Monitoring**: Real-time impact assessment

## Configuration

Key settings in `disaster_response_tool.py`:

```python
CONFIG = {
    'emergency_radius_km': 150,      # Evacuation radius
    'safe_distance_km': 200,        # Minimum safe distance
    'price_surge_factor': 1.2,      # Emergency pricing multiplier
    'worker_relocation_radius': 300, # Worker housing search radius
    'priority_booking_hours': 72,    # Emergency response window
}
```

## Data Requirements

The tool works with standard hotel booking data including:
- Hotel information and location
- Guest demographics and requirements
- Booking patterns and pricing
- Cancellation history
- Seasonal patterns

## Technical Architecture

- **Frontend**: Streamlit dashboard with interactive visualizations
- **Backend**: Python with scikit-learn Random Forest models
- **Visualization**: Plotly for interactive charts and maps
- **Data Processing**: Pandas for efficient data manipulation
- **Caching**: Streamlit caching for optimized performance

## Support and Customization

The tool can be customized for specific:
- Geographic regions and disaster types
- Hotel chains and booking systems
- Integration with existing property management systems
- Custom pricing and availability APIs

## License

Commercial tool for hotel industry disaster response coordination. 