class VaccineData:
    STATIC_VAR_LIST = [
        'Endemic_Potential_R0', 
        'Endemic_Potential_Duration', 
        'Demography_Urban_Rural_Split', 
        'Demography_Population_Density', 
        'Environmental_Index', 
        'Socio_economic_Gini_Index', 
        'Socio_economic_Poverty_Rates',
        'Communication_Affordability',
        'Socio_economic_GDP_per_capita',
        'Socio_economic_Employment_Rates',
        'Socio_economic_Education_Levels'
    ]
    
    TIME_KNOWN_VAR_LIST = [
        'Healthcare_Index_Tier_X_hospitals',
        'Healthcare_Index_Workforce_capacity',
        'Healthcare_Index_Bed_availability_per_capita',
        'Healthcare_Index_Expenditure_per_capita',
        'Immunization_Coverage',
        'Economic_Index_Budget_allocation_per_capita',
        'Economic_Index_Fraction_of_total_budget',
        'Political_Stability_Index'
    ]
    
    TIME_UNKNOWN_VAR_LIST = [
        'Frequency_of_outbreaks',
        'Magnitude_of_outbreaks_Deaths',
        'Magnitude_of_outbreaks_Infected',
        'Magnitude_of_outbreaks_Severity_Index',
        'Security_and_Conflict_Index'
    ]

    FEATURE_OBJECTIVE_MAPPING = {
        'Maximize Health Impact': [
            'Endemic_Potential_R0', 'Endemic_Potential_Duration', 
            'Healthcare_Index_Tier_X_hospitals', 'Healthcare_Index_Workforce_capacity', 
            'Healthcare_Index_Bed_availability_per_capita', 'Immunization_Coverage', 
            'Frequency_of_outbreaks', 'Magnitude_of_outbreaks_Deaths', 
            'Magnitude_of_outbreaks_Infected', 'Magnitude_of_outbreaks_Severity_Index'
        ],
        'Maximize Value for Money': [
            'Economic_Index_Budget_allocation_per_capita', 'Economic_Index_Fraction_of_total_budget',
            'Healthcare_Index_Expenditure_per_capita', 'Socio_economic_GDP_per_capita',
            'Socio_economic_Employment_Rates', 'Socio_economic_Education_Levels'
        ],
        'Reinforce Financial Sustainability': [
            'Economic_Index_Budget_allocation_per_capita', 'Economic_Index_Fraction_of_total_budget',
            'Healthcare_Index_Expenditure_per_capita', 'Socio_economic_GDP_per_capita',
            'Socio_economic_Employment_Rates', 'Socio_economic_Poverty_Rates',
            'Political_Stability_Index', 'Communication_Affordability'
        ],
        'Support Countries with the Greatest Needs': [
            'Demography_Urban_Rural_Split', 'Demography_Population_Density',
            'Environmental_Index', 'Socio_economic_Gini_Index',
            'Socio_economic_Poverty_Rates', 'Healthcare_Index_Bed_availability_per_capita',
            'Political_Stability_Index', 'Security_and_Conflict_Index'
        ]
    }

class DataConfig:
    def __init__(self, STATIC_VAR_LIST, TIME_KNOWN_VAR_LIST, TIME_UNKNOWN_VAR_LIST):
        self.STATIC_VAR_LIST = STATIC_VAR_LIST
        self.TIME_KNOWN_VAR_LIST = TIME_KNOWN_VAR_LIST
        self.TIME_UNKNOWN_VAR_LIST = TIME_UNKNOWN_VAR_LIST
        self.HEALTH_TARGET_VAR = VaccineData.FEATURE_OBJECTIVE_MAPPING['Maximize Health Impact']
        self.VALUE_TARGET_VAR = VaccineData.FEATURE_OBJECTIVE_MAPPING['Maximize Value for Money']
        self.SUSTAINABILITY_TARGET_VAR = VaccineData.FEATURE_OBJECTIVE_MAPPING['Reinforce Financial Sustainability']
        self.NEEDS_TARGET_VAR = VaccineData.FEATURE_OBJECTIVE_MAPPING['Support Countries with the Greatest Needs']

    def get_variable(self, var_name):
        return getattr(self, var_name)
