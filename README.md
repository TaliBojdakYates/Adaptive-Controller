# Adaptive Controller Project

## Overview

This project consists of several scripts and data files related to adaptive control, specifically focusing on activity classification and torque estimation for human movement. The project includes MATLAB scripts for data conversion from .mat to .csv, Python scripts for training and testing activity classification models

## Data Descriptions
This study is based on data from A comprehensive, open-source dataset of lower limb biomechanics in multiple conditions of stairs, ramps, and level-ground ambulation and transitions. This dataset can be found at https://www.epic.gatech.edu/opensource-biomechanics-camargo-et-al/

All the tasks performed in this research was done using python, so for simplicity the files were converted from .mat to .csv. This was done with the `mat_to_csv.mat` file ran in matlab.

All the data can be found in the Dropbox link provided by the dataset. For the training process every trial from patients 6,7,10,11 were used. These can be found in AB{patient}/{date}/{treadmill} and AB{patient}/{date}/{stair}. From these files the imu, knee angle, torque, and force are gotten from imu,ik, id, and fp respectively. In the intial code it was unknown that these were all the categories needed so they were added one at a time as they came up to the data csv files. For future work it is recommended to combine the data from all of these categories into a single csv for each trial instead of one at a time.

For the joint torque profiles patients 6,7,9,10, and 18 were used. Only the first trial from each patient for each of the categories above was used to make the lookup table.


## File Descriptions

### `joint_torque_profile.py`

This Python script calculates normalized knee torque and generates a torque lookup table based on knee angle. It processes data from both stairs and treadmill activities, plots the data, and creates a curve of best fit using spline interpolation.

#### Usage

1.  Ensure that the required libraries (`pandas`, `matplotlib`, `numpy`, `scipy`) are installed.

#### Input

*   `Data/SubjectInfo.csv`: Contains subject information, including weight.
*   `Data/{Subject}/stairs/forces.csv`: Force plate data for stair climbing.
*   `Data/{Subject}/stairs/torques.csv`: Torque data for stair climbing.
*   `Data/{Subject}/stairs/ik.csv`: Inverse kinematics data for stair climbing.
*   `Data/{Subject}/Treadmill/forces.csv`: Force plate data for treadmill walking.
*   `Data/{Subject}/Treadmill/torques.csv`: Torque data for treadmill walking.
*   `Data/{Subject}/Treadmill/ik.csv`: Inverse kinematics data for treadmill walking.

#### Output

*    A CSV file containing the torque lookup table for treadmill and stair walking.


### `mat_to_csv.m`

This MATLAB script converts `.mat` files to `.csv` files. It iterates through a list of `.mat` files, loads the data, and writes it to corresponding `.csv` files.


### `Task Classifier/`

This directory contains scripts and data files for training and testing an activity classifier model.

#### `Task Classifier/train.py`

This Python script trains an LSTM-based activity classifier model using data from CSV files in the "data" folder. It preprocesses the data, creates sequences, splits the data into training and validation sets, trains the model, and evaluates it on a final test dataset.

##### Usage

1.  Ensure that the required libraries (`os`, `pandas`, `numpy`, `torch`, `sklearn`, `matplotlib`, `seaborn`, `tqdm`, `joblib`) are installed.
2.  Place the training data CSV files in the `data/` directory.


##### Input

*   CSV files in the `data/` directory, containing IMU data for stair and treadmill activities. Each csv should be named with either treadmill or stairs

##### Output

*   `activity_classifier_model_final.pth`: A PyTorch model file containing the trained activity classifier model.
*   `imu_scaler.joblib`: A scikit-learn scaler object used for scaling the input data.


#### `Task Classifier/test.py`

This Python script loads a trained activity classifier model, scales the input data, and runs windowed predictions on a test dataset. It then computes the accuracy and torque RMSE. The script 
tests on an unseen patient

##### Usage

1.  Ensure that the required libraries (`os`, `torch`, `joblib`, `pandas`, `numpy`, `sklearn`, `matplotlib`) are installed.
2.  Place the test data CSV file (`final_test_dataset.csv`) in the project directory.


##### Input

*   `activity_classifier_model_final.pth`: A PyTorch model file containing the trained activity classifier model.
*   `imu_scaler.joblib`: A scikit-learn scaler object used for scaling the input data.
*   `final_test_dataset.csv`: A CSV file containing the test data.
*   `torque_lookup_table_stairs.csv`: A CSV file containing the torque lookup table for stair climbing.
*   `torque_lookup_table_treadmill.csv`: A CSV file containing the torque lookup table for treadmill walking.

##### Output

*   Prints the final accuracy and torque RMSE over the test dataset.
*   A plot showing the predicted vs. true knee torque.

#### `Task Classifier/created_test_data.py`

This Python script creates a test dataset from treadmill and stairs data. It reads data from CSV files in the "final\_test" folder, preprocesses the data, and combines it into a single dataset, which is then saved to "final\_test\_dataset.csv". The script extracts features related to IMU, angle , and force plate data.

##### Usage

1.  Ensure that the required libraries (`os`, `pandas`, `numpy`) are installed.

##### Input

*   CSV files in the `final_test/` directory, containing IMU, angle, and force plate data for stair and treadmill activities.

##### Output

*   `final_test_dataset.csv`: A CSV file containing the combined test dataset.


#### `Task Classifier/activity_classifier_model_final.pth`

This file contains the trained activity classifier model.

#### `Task Classifier/imu_scaler.joblib`

This file contains the scaler used to scale the IMU data.

#### `Task Classifier/torque_lookup_table_stairs.csv`

This file contains the torque lookup table for stair climbing.

#### `Task Classifier/torque_lookup_table_treadmill.csv`

This file contains the torque lookup table for treadmill walking.

### `Data/`

This directory contains the data files used for training and testing the activity classifier model.

### `final_test/`

This directory contains the data files used for creating the final test dataset.

## Reproducing Results

To reproduce the results of this project, follow these steps:

1.  Ensure that all required libraries are installed.
2.  Place the data files in the appropriate directories.
3.  Run the scripts in the following order:
    1.  `mat_to_csv.m` (if necessary, to convert `.mat` files to `.csv` files) follow the data section
    2.  `joint_torque_profile.py` (to generate the torque lookup tables)
    3.  `Task Classifier/train.py` (to train the activity classifier model)
    4.  `Task Classifier/created_test_data.py` (to create the final test dataset)
    5.  `Task Classifier/test.py` (to test the activity classifier model)

