"""
Sample Dataset Configurations for Causal AI Platform
Metadata and descriptions for demonstration datasets
"""

# ============================================================================
# SAMPLE DATASET DEFINITIONS
# ============================================================================

SAMPLE_DATASETS = {
    'CO2 SupplyChain Demo.csv': {
        'display_name': '🌱 Supply Chain CO2 Emissions (Logistics Impact Analysis)',
        'description': '''🌱 **Supply Chain CO2 Emissions Analysis**: This dataset examines factors 
                         affecting CO2 emissions in agricultural supply chains. Variables include 
                         transportation method, distance, fuel type, vehicle type, weather conditions, 
                         and resulting emissions.''',
        'goal': 'Identify causal factors that could reduce environmental impact in logistics operations.',
        'key_variables': ['Distance', 'Transport_Mode', 'Vehicle_Type', 'Fuel_Type', 'CO2_Emissions'],
        'expected_insights': [
            'Transportation distance is a primary driver of emissions',
            'Vehicle and fuel type choices significantly impact environmental footprint',
            'Weather conditions may moderate transportation efficiency'
        ],
        'use_cases': [
            'Supply chain optimization',
            'Environmental impact assessment', 
            'Logistics policy planning',
            'Green transportation initiatives'
        ]
    }
}

# ============================================================================
# DATASET CATEGORIES
# ============================================================================

DATASET_CATEGORIES = {
    'environmental': {
        'icon': '🌱',
        'description': 'Environmental impact and sustainability datasets'
    },
    'business': {
        'icon': '💼', 
        'description': 'Business operations and performance datasets'
    },
    'healthcare': {
        'icon': '🏥',
        'description': 'Medical and healthcare outcomes datasets'
    },
    'education': {
        'icon': '🎓',
        'description': 'Educational outcomes and policy datasets'
    }
}

# ============================================================================
# SAMPLE DATA GENERATION PARAMETERS
# ============================================================================

DATA_GENERATION_CONFIG = {
    'default_rows': 1000,
    'noise_level': 0.1,
    'missing_data_rate': 0.05,
    'categorical_levels': {
        'low': 3,      # 3 categories
        'medium': 5,   # 5 categories  
        'high': 8      # 8 categories
    },
    'correlation_ranges': {
        'weak': (0.1, 0.3),
        'moderate': (0.3, 0.6),
        'strong': (0.6, 0.9)
    }
}
