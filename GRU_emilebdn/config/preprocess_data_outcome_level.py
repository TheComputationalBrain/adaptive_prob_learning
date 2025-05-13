# Created on 2025/04/28 - @emilebdn

#%%
import os.path as op
import pandas as pd

from datetime import datetime
from paths import computed_data_emile_path, data_outcome_level_path

# Dynamically generate today's date
today_date = datetime.now().strftime('%Y%m%d')

data_outcome_level = pd.read_csv(data_outcome_level_path)

# Exclude rows where subject is '604b169fe4b7991ec08da3a6'
data_outcome_level_preprocessed = data_outcome_level[data_outcome_level['subject'] != '604b169fe4b7991ec08da3a6']

# Save preprocessed data 
data_outcome_level_preprocessed.to_csv(op.join(computed_data_emile_path, f'{today_date}_data_outcome_level_preprocessed.csv'), index=False)

# %%
