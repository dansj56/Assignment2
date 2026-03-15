#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

#%%
trail1 = pd.read_csv('Trail1_extracted_features_acceleration_m1ai1-1-1.csv')
trail2 = pd.read_csv('Trail2_extracted_features_acceleration_m1ai1-2.csv')
trail3 = pd.read_csv('Trail3_extracted_features_acceleration_m2ai0-2.csv')
# print(trail1.info())
# print(trail2.info())
# print(trail3.info())
# print(f'Trail1 shape: {trail1.shape}')
# print(f'Trail2 shape: {trail2.shape}')
# print(f'Trail3 shape: {trail3.shape}')

#%%
# Combine the three files and remove columns
trails = [trail1,trail2,trail3]
traildata = pd.concat(trails, ignore_index=True).drop(columns=['start_time','axle','cluster','tsne_1','tsne_2'])
print(f'Traildata shape: {traildata.shape}')

# %%
#Replacing all 'normal' events with 0, all others with 1
#Separating target and features
# print(traildata['event'].unique())
# print(f'Before, sum of normal events: {(traildata['event'] == 'normal').sum()}')
# print(f'Before, sum of anomalies: {(traildata['event'] != 'normal').sum()}')
traildata['event'] = (traildata['event'] != 'normal').astype(int)
# print(f'After, sum of normal events: {(traildata['event'] == 0).sum()}')
# print(f'After, sum of anomalies: {(traildata['event'] != 0).sum()}')
# print(traildata.info())
trailtarget = traildata['event']
print("Classes in training set:", np.unique(trailtarget))
trail = traildata.drop(columns='event')

# %%
#Splitting into training and testing data sets: 80/20
print("Class distribution in full data:")
print(trailtarget.value_counts())
trail_train, trail_test, target_train, target_test = (
    train_test_split(trail, trailtarget, test_size=0.2, random_state=42, stratify=trailtarget)
)
print(f'Trail train shape: {trail_train.shape}, Trail test shape: {trail_test.shape}')
print(f'Target train shape: {target_train.shape}, Target test shape: {target_test.shape}')
# %%
#Normalize features
#traildata.isnull().sum() #No null data in columns!
scaler = StandardScaler()
trail_train_scaled = pd.DataFrame(
    scaler.fit_transform(trail_train),
    columns=trail_train.columns
)
#Scale the test set based on the scaling from the training set.
trail_test_scaled = pd.DataFrame(
    scaler.transform(trail_test),
    columns=trail_test.columns
)
print(f'Before scaling - Mean: {trail.mean().mean():.4f}, Std: {trail.std().mean():.4f}')
print(f'After scaling (Training set) - Mean: {trail_train_scaled.mean().mean():.4f}, Std: {trail_train_scaled.std().mean():.4f}')

#%%
#Train SVM-model
model = SVC(gamma='auto', random_state=42)
model.fit(trail_train_scaled,target_train)

# %%
#Use model to predict on the test set
predictions = model.predict(trail_test_scaled)

#%%
#Evaluate results
print(f'Classification report: \n {classification_report(target_test, predictions, target_names=["Normal", "Anomaly"])}')
print(f'Accuracy score: {accuracy_score(target_test, predictions):.5f}')

# %%
#Using KFold cross validation on the test set only.
#kf = KFold(n_splits=5)
scores = cross_val_score(model, trail_train_scaled, target_train, cv=5)
#print(f'Scores from KFold:  {scores}')
print(f'Mean of scores: {scores.mean():.5f}')
print(f'Standard deviation of scores: {scores.std():.5f}')

# %%
#Using KFold cross validation on the ENTIRE set.
scaler_full = StandardScaler()
trail_scaled_full = scaler_full.fit_transform(trail)
scores_full = cross_val_score(model, trail_scaled_full, trailtarget, cv=5)
print(f"K-Fold Entire dataset Mean:  {scores_full.mean():.5f}")
print(f"K-Fold Std Deviation:   {scores_full.std():.5f}")

#%%