import pandas as pd
import numpy as np
import math
from random import randrange

# Get path to data folder
import os
data_path = os.path.join(os.path.split(os.getcwd())[0], 'data')

# Load specific data
drop_columns = ['DefendersInTheBox_vs_Distance','HomePossesion','Field_eq_Possession','HomeField','Formation_ACE','Formation_EMPTY','Formation_I_FORM','Formation_JUMBO','Formation_PISTOL','Formation_SHOTGUN','Formation_SINGLEBACK','Formation_WILDCAT','PlayerBMI','PlayerAge','YardsLeft']
df = pd.read_csv(os.path.join(data_path, 'test_data.csv'), index_col=0).drop(columns=drop_columns)

# Standardize coordinates
df['HomeOnOffense'] = df['PossessionTeam'] == df['HomeTeamAbbr']
df['Offense'] = df['HomeOnOffense'] & df['Team'] | ~df['HomeOnOffense'] & ~df['Team']

#df['X'] = np.where(~df['PlayDirection'], df['X'], 120-df['X'])
#df['Y'] = np.where(~df['PlayDirection'], df['Y'], 53.3-df['Y'])

df['YardLine'] = np.where(df['FieldPosition'] == df['PossessionTeam'], 100-df['YardLine'], df['YardLine'])

# Aggregate plays
plays_df = df.groupby('PlayId')

# Write random play to json
n = randrange(len(plays_df.groups))
play = plays_df.get_group(list(plays_df.groups)[n])

filename = 'play.json'
play.to_json(filename, 'records')

print(f'Loaded new play with ID {n}')