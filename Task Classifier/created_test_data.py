'''
This script creates a test dataset from treadmill and stairs data.
It reads data from CSV files in the "final_test" folder, preprocesses the data,
and combines it into a single dataset, which is then saved to "final_test_dataset.csv".
The script extracts features related to IMU, goniometer, and force plate data.
'''
import os
import pandas as pd
import numpy as np

DATA_FOLDER = "final_test"  
FEATURES__TREADMILL = ['imu_thigh_Gyro_X', 'imu_thigh_Gyro_Y', 'imu_thigh_Gyro_Z',
                'imu_thigh_Accel_X', 'imu_thigh_Accel_Y', 'imu_thigh_Accel_Z', 'gon_knee_sagittal', 'knee_angle_r_moment', 'Treadmill_L_vz']

FEATURES__STAIRS = ['imu_thigh_Gyro_X', 'imu_thigh_Gyro_Y', 'imu_thigh_Gyro_Z',
                'imu_thigh_Accel_X', 'imu_thigh_Accel_Y', 'imu_thigh_Accel_Z', 'gon_knee_sagittal', 'knee_angle_r_moment', 'FP1_vy','FP2_vy','FP3_vy','FP4_vy']


files = sorted([f for f in os.listdir(DATA_FOLDER) if f.endswith('.csv')])

treadmill_file = next(f for f in files if 'treadmill' in f.lower())
stairs_files = [f for f in files if 'stair' in f.lower()]

treadmill_df = pd.read_csv(os.path.join(DATA_FOLDER, treadmill_file))[FEATURES__TREADMILL]
condition_2 = (abs(treadmill_df['Treadmill_L_vz']) >= 0.1)
treadmill_df['phase'] = condition_2.astype(int)

treadmill_label = np.zeros(len(treadmill_df))
treadmill_parts = np.array_split(treadmill_df, 5)
treadmill_label_parts = np.array_split(treadmill_label, 5)

# Load 1/3 of each stairs file
stairs_dfs = []
stairs_labels = []
for sf in stairs_files:
    df = pd.read_csv(os.path.join(DATA_FOLDER, sf))[FEATURES__STAIRS]
    print(df.columns.tolist())
    condition = (df['FP1_vy'] != 0) | (df['FP2_vy'] != 0) | (df['FP3_vy'] != 0) | (df['FP4_vy'] != 0)

    df['phase'] = condition.astype(int)
    
    cut = len(df) // 3
    stairs_dfs.append(df[:cut])
    stairs_labels.append(np.ones(cut))  


combined_X = []
combined_y = []
for i in range(4):
    combined_X.append(treadmill_parts[i])
    combined_y.append(pd.Series(treadmill_label_parts[i]))
    combined_X.append(stairs_dfs[i])
    combined_y.append(pd.Series(stairs_labels[i]))


combined_X.append(treadmill_parts[-1])
combined_y.append(pd.Series(treadmill_label_parts[-1]))

X = pd.concat(combined_X, ignore_index=True)
y = pd.concat(combined_y, ignore_index=True)

X["label"] = y.astype(int)


X.to_csv("final_test_dataset.csv", index=False)
print("Saved dataset to 'final_test_dataset.csv'")
