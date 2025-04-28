'''
This script loads a trained activity classifier model, scales the input data,
and runs windowed predictions on a test dataset. It then computes the accuracy
and torque RMSE. The script also plots the predicted vs true knee torque.
'''
import os
import torch
import joblib 
import pandas as pd
import numpy as np
from train import ActivityClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, root_mean_squared_error
import matplotlib.pyplot as plt

# Configuration
WINDOW_SIZE = 100
MODEL_SAVE_PATH = 'activity_classifier_model_final.pth'
SCALER_PATH = 'classification_results_final_test/imu_scaler.joblib'
IMU_FEATURES = ['imu_thigh_Gyro_X', 'imu_thigh_Gyro_Y', 'imu_thigh_Gyro_Z',
                'imu_thigh_Accel_X', 'imu_thigh_Accel_Y', 'imu_thigh_Accel_Z']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AB8_WEIGHT = 72.57

df_stairs = pd.read_csv('C:/Users/talib/Desktop/Biotics/Task Classifier/torque_lookup_table_stairs.csv')
df_treadmill = pd.read_csv('C:/Users/talib/Desktop/Biotics/Task Classifier/torque_lookup_table_treadmill.csv')
lookup_stairs = dict(zip(df_stairs["Knee Angle (deg)"], df_stairs["Torque (Nm/kg)"]))
lookup_treadmill = dict(zip(df_treadmill["Knee Angle (deg)"], df_treadmill["Torque (Nm/kg)"]))

input_size = len(IMU_FEATURES)
model = ActivityClassifier(input_size=input_size, hidden_size=128, num_layers=2, num_classes=2)
model.load_state_dict(torch.load(MODEL_SAVE_PATH))
model.to(DEVICE)
model.eval()


scaler = joblib.load(SCALER_PATH)


csv_path = 'final_test_dataset.csv'
df = pd.read_csv(csv_path)
angles = df['gon_knee_sagittal']
true_torques = df['knee_angle_r_moment']
phase = df['phase']
required_columns = IMU_FEATURES + ['label']

X_raw = df[IMU_FEATURES].values
y_true_all = df['label'].values

X_scaled = scaler.transform(X_raw)

all_preds = []
all_true = []
pred_torque_list = []
true_torque_list = []
print("\n--- Running Windowed Predictions ---")
for i in range(0, len(X_scaled) - WINDOW_SIZE + 1,WINDOW_SIZE):
    window = X_scaled[i:i + WINDOW_SIZE]
    window_tensor = torch.FloatTensor(window).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        output = model(window_tensor)
        pred_class = torch.argmax(output, dim=1).item()

    true_label = int(np.round(np.mean(y_true_all[i:i + WINDOW_SIZE])))

    print(f"Window {i}-{i+WINDOW_SIZE}: Predicted = {pred_class}, True = {true_label}")
    mid_idx = i + WINDOW_SIZE // 2
    current_angle = angles[mid_idx]
    current_torque = true_torques[mid_idx]
    
    # 0: level ground, 1: stairs
    # Phase 0: swing 1:stance
    if phase[i]:
        try:
            if pred_class:
                norm_pred_torque = lookup_stairs.get(round(current_angle), np.nan)
            else:
                norm_pred_torque = lookup_treadmill.get(round(current_angle), np.nan)
            
            if not np.isnan(norm_pred_torque) and not np.isnan(current_torque):
                pred_torque = AB8_WEIGHT * norm_pred_torque
                pred_torque_list.append(pred_torque)
                true_torque_list.append(current_torque)
        except  Exception as e:
            print("FAILED")
            print(e)
    
    all_preds.append(pred_class)
    all_true.append(true_label)

accuracy = accuracy_score(all_true, all_preds)
print(f"\nFinal Accuracy over {len(all_preds)} windows: {accuracy:.4f}")

torque_error = root_mean_squared_error(true_torque_list,pred_torque_list)
print(f"\nFinal Torque RMSE over {len(all_preds)} windows: {torque_error:.4f}")

plt.figure(figsize=(12, 6))
plt.plot(true_torque_list[15:40], label='True Torque', linewidth=2)
plt.plot(pred_torque_list[15:40], label='Predicted Torque', linewidth=2, linestyle='--')
plt.title('Predicted vs True Knee Torque')
plt.xlabel('Window Index')
plt.ylabel('Torque (Nm)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()