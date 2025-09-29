import os
import sys

# Ensure project root (which contains `utils/`) is on path
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_THIS_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.config import VaccineData as _VaccineData


class FeatureConfig:
    STATIC = list(_VaccineData.STATIC_VAR_LIST)
    KNOWN = list(_VaccineData.TIME_KNOWN_VAR_LIST)
    UNKNOWN = list(_VaccineData.TIME_UNKNOWN_VAR_LIST)

    OBJECTIVES = {
        'Maximize Health Impact': 'target__Magnitude_of_outbreaks_Infected',
        'Maximize Value for Money': 'target__Magnitude_of_outbreaks_Deaths',
        'Reinforce Financial Sustainability': 'target__Magnitude_of_outbreaks_Severity_Index',
        'Support Countries with the Greatest Needs': 'target__Frequency_of_outbreaks',
    }

    FINAL_WEIGHTS = {
        'Maximize Health Impact': 0.30,
        'Maximize Value for Money': 0.30,
        'Reinforce Financial Sustainability': 0.25,
        'Support Countries with the Greatest Needs': 0.15,
    }


