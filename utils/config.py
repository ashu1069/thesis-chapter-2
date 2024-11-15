class VaccinePrioritization:
    def __init__(self):
        self.static_var_list = [
            'Transmission Mode', 'Endemic Potential R0', 'Endemic Potential CFR',
            'Endemic Potential Duration', 'Demography Urban-Rural Split',
            'Demography Population Density', 'Environmental Index',
            'Communication Affordability', 'Communication Media Freedom',
            'Communication Connectivity', 'Socio-economic GDP per capita',
            'Socio-economic Gini Index', 'Socio-economic Employment Rates',
            'Socio-economic Poverty Rates', 'Socio-economic Education Levels'
        ]

        self.time_known_var_list = [
            'Healthcare Index Tier-X hospitals', 'Healthcare Index Workforce capacity',
            'Healthcare Index Bed availability per capita', 'Healthcare Index Expenditure per capita',
            'Political Stability Index', 'Economic Index Budget allocation per capita',
            'Economic Index Fraction of total budget', 'Immunization Coverage'
        ]

        self.time_unknown_var_list = [
            'Frequency of outbreaks', 'Magnitude of outbreaks Deaths',
            'Magnitude of outbreaks Infected', 'Magnitude of outbreaks Severity Index',
            'Security and Conflict Index'
        ]

        self.objectives = {
            'Maximize Health Impact': 0.30,
            'Maximize Value for Money': 0.30,
            'Reinforce Financial Sustainability': 0.25,
            'Support Countries with the Greatest Needs': 0.15,
            'Promote Equitable Distribution': None  # static importance among all
        }

        self.indicator_mapping = self._map_indicators()

    def _map_indicators(self):
        return {
            'Maximize Health Impact': [
                'Endemic Potential R0', 'Endemic Potential CFR', 'Endemic Potential Duration',
                'Healthcare Index Tier-X hospitals', 'Healthcare Index Workforce capacity',
                'Healthcare Index Bed availability per capita', 'Immunization Coverage',
                'Frequency of outbreaks', 'Magnitude of outbreaks Deaths',
                'Magnitude of outbreaks Infected', 'Magnitude of outbreaks Severity Index'
            ],
            'Maximize Value for Money': [
                'Economic Index Budget allocation per capita', 'Economic Index Fraction of total budget',
                'Healthcare Index Expenditure per capita', 'Socio-economic GDP per capita',
                'Socio-economic Employment Rates', 'Socio-economic Education Levels'
            ],
            'Reinforce Financial Sustainability': [
                'Economic Index Budget allocation per capita', 'Economic Index Fraction of total budget',
                'Healthcare Index Expenditure per capita', 'Socio-economic GDP per capita',
                'Socio-economic Employment Rates', 'Socio-economic Poverty Rates',
                'Political Stability Index', 'Communication Affordability'
            ],
            'Support Countries with the Greatest Needs': [
                'Demography Urban-Rural Split', 'Demography Population Density', 'Environmental Index',
                'Socio-economic Gini Index', 'Socio-economic Poverty Rates',
                'Healthcare Index Bed availability per capita', 'Political Stability Index',
                'Security and Conflict Index'
            ],
            'Promote Equitable Distribution': self.static_var_list + self.time_known_var_list + self.time_unknown_var_list
        }

    def get_indicators_by_objective(self, objective):
        return self.indicator_mapping.get(objective, [])

    def get_objective_weight(self, objective):
        return self.objectives.get(objective, None)

# # Example usage
# vp = VaccinePrioritization()

# # Get indicators for "Maximize Health Impact"
# health_impact_indicators = vp.get_indicators_by_objective('Maximize Health Impact')
# print("Health Impact Indicators:", health_impact_indicators)

# # Get weight for "Maximize Value for Money"
# value_for_money_weight = vp.get_objective_weight('Maximize Value for Money')
# print("Value for Money Weight:", value_for_money_weight)

class DataConfig:
    def __init__(self, **kwargs):
        self.config = kwargs

    def add_variable(self, name, value):
        self.config[name] = value
    
    def get_variable(self, name):
        return self.config.get(name, None)
    
    def remove_variable(self, name):
        if name in self.config:
            del self.config[name]

    def update_variable(self, name , value):
        if name in self.config:
            self.config[name] = value
        else:
            raise KeyError(f"Variable '{name}' not found in the configuration")
        
    def get_all_variables(self):
        return self.config
    
    def __repr__(self):
        return f"DataConfig({self.config})"
    

class VaccineData:
    # ... existing variables ...
    HEALTH_TARGET_VAR = 'health_impact'
    VALUE_TARGET_VAR = 'value_for_money'
    SUSTAINABILITY_TARGET_VAR = 'financial_sustainability'
    NEEDS_TARGET_VAR = 'country_needs'
    EQUITY_TARGET_VAR = 'equity_score'
