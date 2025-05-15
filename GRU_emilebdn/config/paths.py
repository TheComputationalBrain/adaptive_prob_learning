"""
This script defines paths for accessing and preprocessing outcome-level data, 
accessing already completed analyses and simulation data.

Author: @emilebdn  
Created date: 2025-03-22
"""

import os
import os.path as op

computer = os.uname()[1]
home_path = op.expanduser("~")

project_path = op.join(home_path, 'nasShare', 'projects', 'protocols', 'AdaptiveProbLearning_ChungMeyniel_2025')
data_path = op.join(project_path, \
                    'data', 'Foucault_Meyniel_2024')

ada_learn_study_path = op.join(data_path, 'structured-dataset', 'ada-learn_study')
data_outcome_level_path = op.join(ada_learn_study_path, 'data_outcome-level.csv')

computed_data_emile_path = op.join(data_path, 'computed_data_emile')
data_outcome_level_preprocessed_path = op.join(computed_data_emile_path, '20250506_data_outcome_level_preprocessed.csv')

data_outcome_level_simulated_path = op.join(computed_data_emile_path, '20250513_data_outcome_level_simulated.csv')