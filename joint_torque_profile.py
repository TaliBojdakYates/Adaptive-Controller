'''
This script calculates normalized knee torque,and generates a torque lookup table
based on knee angle. It processing data from both stairs and treadmill activities,
plotting the data and creating a curve of best fit using spline interpolation.
'''
import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from scipy.interpolate import UnivariateSpline

def process_data_stairs(directory):
    subjectInfo = pd.read_csv(os.path.join(directory, 'SubjectInfo.csv'))
    combined_ascending = pd.DataFrame()

    for foldername in os.listdir(directory):
        folder_path = Path(directory) / foldername / 'stairs'

        if foldername.startswith('AB') and folder_path.exists():
            subject = foldername
            if subject not in subjectInfo['Subject'].values:
                continue

            weight = subjectInfo[subjectInfo['Subject'] == subject]['Weight'].values[0]
            print(f'Processing {subject}, Weight: {weight}')

            files = {
                'forces': folder_path / 'forces.csv',
                'torques': folder_path / 'torques.csv',
                'ik': folder_path / 'ik.csv'
            }

            if all(f.exists() for f in files.values()):
                df_forces = pd.read_csv(files['forces'])[['Header', 'FP2_vy', 'FP3_vy', 'FP4_vy']]
                df_valid_forces = df_forces.loc[
                    (df_forces['FP2_vy'] != 0) | 
                    (df_forces['FP3_vy'] != 0) | 
                    (df_forces['FP4_vy'] != 0)
                ]

                df_torques = pd.read_csv(files['torques'])[['Header', 'knee_angle_l_moment']]
                df_torques['knee_angle_l_moment'] = df_torques['knee_angle_l_moment'] / weight

                df_angles = pd.read_csv(files['ik'])[['Header', 'knee_angle_l']]

                left_leg_df = df_valid_forces.merge(df_torques, on='Header')\
                                             .merge(df_angles, on='Header')\
                                             [['Header', 'knee_angle_l_moment', 'knee_angle_l']]
                
                left_leg_df = left_leg_df.iloc[:len(left_leg_df) // 3]

                combined_ascending = pd.concat([combined_ascending, left_leg_df], ignore_index=True)

    return combined_ascending.dropna()
    

def process_data_treadmill(directory):
    subjectInfo = pd.read_csv(os.path.join(directory, 'SubjectInfo.csv'))
    combined_ascending = pd.DataFrame()

    for foldername in os.listdir(directory):
        folder_path = Path(directory) / foldername / 'Treadmill'

        if foldername.startswith('AB') and folder_path.exists():
            subject = foldername
            if subject not in subjectInfo['Subject'].values:
                continue

            weight = subjectInfo[subjectInfo['Subject'] == subject]['Weight'].values[0]
            print(f'Processing {subject}, Weight: {weight}')

            files = {
                'forces': folder_path / 'forces.csv',
                'torques': folder_path / 'torques.csv',
                'ik': folder_path / 'ik.csv'
            }

            if all(f.exists() for f in files.values()):
                df_forces = pd.read_csv(files['forces'])[['Header', 'Treadmill_L_vz']]
                
                non_zero_moments_df = df_forces[abs(df_forces['Treadmill_L_vz']) >= 0.1]
               
                df_torques = pd.read_csv(files['torques'])[['Header', 'knee_angle_l_moment']]
                df_torques['knee_angle_l_moment'] = df_torques['knee_angle_l_moment'] / weight

                df_angles = pd.read_csv(files['ik'])[['Header', 'knee_angle_l']]

                left_leg_df = non_zero_moments_df.merge(df_torques, on='Header')\
                                             .merge(df_angles, on='Header')\
                                             [['Header', 'knee_angle_l_moment', 'knee_angle_l']]      
               

                combined_ascending = pd.concat([combined_ascending, left_leg_df], ignore_index=True)

    return combined_ascending.dropna()



def plot_data(climbing_df,path):

    fig, ax1 = plt.subplots(figsize=(10, 6))

   
    ax1.scatter(climbing_df['knee_angle_l'], 
                           climbing_df['knee_angle_l_moment'],
                           
                           label='Raw Data')

  
 
    x, y = curve_of_best_fit(climbing_df, path)
    ax1.plot(x, y, 'r-', linewidth=3, label='Spline Fit')

    ax1.set_title('Torque/Weight vs Knee Angle')
    ax1.set_xlabel('Knee Sagittal Angle (degrees)')
    ax1.set_ylabel('Normalized Knee Torque (Nm/kg)')

    ax1.legend()

    plt.tight_layout()
    plt.show()



def curve_of_best_fit(df, path, num_points=100):
    df_sorted = df.sort_values('knee_angle_l')

    # Create bins
    bins = np.linspace(df_sorted['knee_angle_l'].min(), 
                       df_sorted['knee_angle_l'].max(), 
                       100)
    df_sorted['angle_bin'] = pd.cut(df_sorted['knee_angle_l'], bins)

    df_binned = df_sorted.groupby('angle_bin')['knee_angle_l_moment'].mean().reset_index()

    df_binned = df_binned.dropna()

    bin_centers = [(interval.left + interval.right)/2 for interval in df_binned['angle_bin']]

    spline = UnivariateSpline(bin_centers, df_binned['knee_angle_l_moment'], s=1)

    x_smooth = np.linspace(min(bin_centers), max(bin_centers), num_points)
    y_smooth = spline(x_smooth)

    lookup_angles = np.arange(np.ceil(min(bin_centers)), np.floor(max(bin_centers)) + 1, 1)
    lookup_torques = spline(lookup_angles)

    lookup_table = pd.DataFrame({
        'Knee Angle (deg)': lookup_angles,
        'Torque (Nm/kg)': lookup_torques
    })
    lookup_table.to_csv(path, index=False)

    return x_smooth, y_smooth

def main():
    directory = "Data"
    #climbing_df = process_data_stairs(directory)
    climbing_df = process_data_treadmill(directory)
    path = 'torque_lookup_table_treadmill.csv'
    plot_data(climbing_df,path)

if __name__ == "__main__":
    main()
