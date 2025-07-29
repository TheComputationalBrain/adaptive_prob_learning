"""
This script defines paths for accessing and preprocessing outcome-level data, 
and accessing already completed analyses.

Author: @emilebdn  
Created date: 2025-03-22
"""
import os.path as op

root_dir = op.dirname(op.dirname(op.dirname(__file__)))

data_outcome_level_path = op.join(root_dir, 'data', 'Foucault_Meyniel_2024', \
                    'structured-dataset', 'ada-learn_study', 'data_outcome-level.csv')

computed_data_emile_path = op.normpath(
    op.join(
        root_dir, '..', '..', 'ebayondenoyer', 'nasShare', \
        'projects', 'protocols', 'AdaptiveProbLearning_ChungMeyniel_2025', \
        'data', 'Foucault_Meyniel_2024', 'computed_data_emile'
    )
)

data_outcome_level_preprocessed_path = op.join(computed_data_emile_path, 'data_outcome_level', \
                                               '20250515_data_outcome_level_preprocessed.csv')