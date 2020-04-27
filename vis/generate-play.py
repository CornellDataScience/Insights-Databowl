import pandas as pd
import math
from random import randrange

# Get path to data folder
import os
data_path = os.path.join(os.path.split(os.getcwd())[0], 'data')

# Aggregate and filter play data
drop_columns = ['GameId','DefendersInTheBox_vs_Distance','HomePossesion','Field_eq_Possession','HomeField','Formation_ACE','Formation_EMPTY','Formation_I_FORM','Formation_JUMBO','Formation_PISTOL','Formation_SHOTGUN','Formation_SINGLEBACK','Formation_WILDCAT','PlayerBMI','TimeDelta','PlayerAge','YardsLeft']
df = pd.read_csv(os.path.join(data_path, 'test_data.csv'), index_col=0).drop(columns=drop_columns)
plays_df = df.groupby('PlayId')

# Select random play
n = randrange(len(plays_df.groups))
play = plays_df.get_group(list(plays_df.groups)[n])

# Write play to JSON file
filename = 'play.json'
play.to_json(filename, 'records')