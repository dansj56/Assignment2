#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%%
trail1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1-1.csv')
trail2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1-2.csv')
trail3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0-2.csv')
#print(trail1.info())
#print(trail2.info())
#print(trail3.info())

#%%
# Combine the three files and remove columns
trails = [trail1,trail2,trail3]
traildata = pd.concat(trails).drop(columns=['start_time','cluster','tsne_1','tsne_2'])
#print(traildata.info())

# %%
#Replacing a 'normal' events with 0, all others with 1
print(traildata['event'].unique())
traildata['event'] = (traildata['event'] != 'normal').astype(int)
print(traildata['event'].unique())
print(traildata.info())
# %%
