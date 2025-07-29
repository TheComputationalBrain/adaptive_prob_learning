"""
This script preprocesses outcome-level data by excluding specific subjects 
and saves the cleaned dataset for further analysis. 

Author: @emilebdn  
Created date: 2025-04-28
"""
import sys

import os.path as op
import pandas as pd

from datetime import datetime

sys.path.append(op.dirname(op.dirname(op.dirname(__file__))))

from emilebdn.config.paths import (
    computed_data_emile_path, 
    data_outcome_level_path
)

today = datetime.now().strftime('%Y%m%d')

# Load the initial data
data_outcome_level = pd.read_csv(data_outcome_level_path)

# Exclude rows where subject is '604b169fe4b7991ec08da3a6'
# This subject has multiple data for the same sequence
data_outcome_level_preprocessed = \
    data_outcome_level[data_outcome_level['subject'] != '604b169fe4b7991ec08da3a6']

# Drop the first column which is an index column
data_outcome_level_preprocessed = data_outcome_level_preprocessed.iloc[:, 1:]

# Save preprocessed data 
data_outcome_level_preprocessed.to_csv(
    op.join(computed_data_emile_path, \
            f'{today}_data_outcome_level_preprocessed.csv'), index=False
)