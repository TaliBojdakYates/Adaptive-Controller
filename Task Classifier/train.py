'''
This script trains an LSTM-based activity classifier model using data from CSV files
in the "data" folder. It preprocesses the data, creates sequences, splits the data
into training and validation sets, trains the model, and evaluates it on a final
test dataset. 
'''
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 
from tqdm.auto import tqdm
import joblib 

DATA_FOLDER = 'data' 
TEST_DATA_FOLDER = 'test_data' #

WINDOW_SIZE = 100
NUM_CLASSES = 2     # 0: level ground, 1: stairs
HIDDEN_SIZE = 128
NUM_LAYERS = 2
BATCH_SIZE = 128
N_EPOCHS = 50
LEARNING_RATE = 0.001
MODEL_SAVE_PATH = 'activity_classifier_model_final.pth' 
RESULTS_FOLDER = 'classification_results_final_test'

SCALER_SAVE_PATH = 'imu_scaler.joblib'


IMU_FEATURES = ['imu_thigh_Gyro_X', 'imu_thigh_Gyro_Y', 'imu_thigh_Gyro_Z', 'imu_thigh_Accel_X','imu_thigh_Accel_Y','imu_thigh_Accel_Z']


if not os.path.exists(RESULTS_FOLDER):
    os.makedirs(RESULTS_FOLDER)
    print(f"Created results folder: {RESULTS_FOLDER}")


class ActivityClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super(ActivityClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
            batch_first=True, dropout=0.2 if num_layers > 1 else 0
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2), nn.ReLU(),
            nn.Dropout(0.3), nn.Linear(hidden_size // 2, num_classes)
        )
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        lstm_out, _ = self.lstm(x, (h0, c0))
        out = self.fc(lstm_out[:, -1, :])
        return out


class GaitActivityDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
    def __len__(self):
        return len(self.y)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def create_sequences(X, y, window_size=WINDOW_SIZE, step_size=1, desc="Creating sequences"):
    Xs, ys = [], []   
    print(f"{desc}...")
    if len(X) <= window_size:
         print(f"Warning: Data length ({len(X)}) is not greater than window size ({window_size}). Cannot create sequences.")
         return np.array([]), np.array([])
    
    for i in tqdm(range(0, len(X) - window_size, step_size), desc=desc):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size]) 

    if not Xs:
        return np.array([]), np.array([])
    
    return np.array(Xs), np.array(ys)


if __name__ == "__main__":
    print(f"\n--- Processing Training/Validation Data from: {DATA_FOLDER} ---")
    all_train_val_X = []
    all_train_val_y = []
    train_val_files = 0
    train_val_file_list = [f for f in os.listdir(DATA_FOLDER) if f.lower().endswith('.csv')]

    if not train_val_file_list:
        print(f"Error: No CSV files")
        exit()

    for filename in tqdm(train_val_file_list, desc=f"Loading Train/Val CSVs from {DATA_FOLDER}"):
        file_path = os.path.join(DATA_FOLDER, filename)
        try:
            df = pd.read_csv(file_path)
            if not all(col in df.columns for col in IMU_FEATURES):
                tqdm.write(f"Warning Skipping {filename}. Missing columns.")
                continue

            label = -1
            if 'stair' in filename.lower():
                label = 1
                df = df.iloc[:len(df) // 3]
            elif 'treadmill' in filename.lower(): label = 0

            if label != -1:
            
                if len(df) > WINDOW_SIZE:
                    train_val_files += 1
                    file_X = df[IMU_FEATURES].values
                    file_y = np.full(len(file_X), label)
                    all_train_val_X.append(file_X)
                    all_train_val_y.append(file_y)
                else:
                    tqdm.write(f"Warning Skipping {filename}. Insufficient rows.")


        except pd.errors.EmptyDataError:
            tqdm.write(f"Warning empty file {filename}.")
        except Exception as e:
            tqdm.write(f"Error processing {filename}: {e}")

    if not all_train_val_X:
        print("Error: No valid training/validation data")
        exit()


    X_train_val_concat = np.concatenate(all_train_val_X, axis=0)
    y_train_val_concat = np.concatenate(all_train_val_y, axis=0)
    del all_train_val_X, all_train_val_y # Free memory since computer was running out


    print("Fitting scaler on training/validation data")
    scaler_X = StandardScaler()
    X_train_val_scaled = scaler_X.fit_transform(X_train_val_concat)
    del X_train_val_concat # Free memory since computer was running out 
    print("Scaler fitting complete.")

    joblib.dump(scaler_X, SCALER_SAVE_PATH)
    print("Scaler saved.")


    X_train_val_seq, y_train_val_seq = create_sequences(
        X_train_val_scaled, y_train_val_concat, window_size=WINDOW_SIZE, step_size=WINDOW_SIZE, desc="Creating Train/Val sequences"
    )
    del X_train_val_scaled, y_train_val_concat # Free memory

    if X_train_val_seq.size == 0:
        print("Error: check data and window size.")
        exit()

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val_seq, y_train_val_seq,
            test_size=0.20, 
            random_state=42,
            stratify=y_train_val_seq
        )
    except ValueError as e:
       
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val_seq, y_train_val_seq, test_size=0.20, random_state=42
        )
    del X_train_val_seq, y_train_val_seq # Free memory

    print(f"\nTraining set size: {len(X_train)}")
    print(f"Validation set size: {len(X_val)}")

 
    train_dataset = GaitActivityDataset(X_train, y_train)
    val_dataset = GaitActivityDataset(X_val, y_val)
    del X_train, y_train, X_val, y_val # Free memory

    num_workers = 0 
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")

    input_size = len(IMU_FEATURES)
    model = ActivityClassifier(input_size, HIDDEN_SIZE, NUM_LAYERS, NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n--- Starting Training ---")
    train_losses = []
    val_losses = []
    best_val_accuracy = 0.0 

    for epoch in range(N_EPOCHS):
        model.train()
        epoch_train_loss = 0
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Train]", unit="batch")
        for batch_X, batch_y in train_iterator:
            batch_X, batch_y = batch_X.to(device, non_blocking=pin_memory), batch_y.to(device, non_blocking=pin_memory)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            current_loss = loss.item()
            epoch_train_loss += current_loss
            train_iterator.set_postfix(loss=f"{current_loss:.4f}")


        model.eval()
        epoch_val_loss = 0
        all_val_preds = []
        all_val_labels = []
        val_iterator = tqdm(val_loader, desc=f"Epoch {epoch+1}/{N_EPOCHS} [Validate]", unit="batch", leave=False)
        with torch.no_grad():
            for batch_X, batch_y in val_iterator:
                batch_X, batch_y = batch_X.to(device, non_blocking=pin_memory), batch_y.to(device, non_blocking=pin_memory)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                epoch_val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                all_val_preds.extend(predicted.cpu().numpy())
                all_val_labels.extend(batch_y.cpu().numpy())
                val_iterator.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = epoch_train_loss / len(train_loader)
        avg_val_loss = epoch_val_loss / len(val_loader) 
        val_accuracy = accuracy_score(all_val_labels, all_val_preds) 

        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss) 
        print(f'Epoch [{epoch+1:02d}/{N_EPOCHS}] Summary -> Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')

    
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"  -> Saved new best model to {MODEL_SAVE_PATH} (Validation Accuracy: {best_val_accuracy:.4f})")

    print("\n--- Training Finished ---")
    print(f"Best Validation Accuracy: {best_val_accuracy:.4f}")


    plt.figure(figsize=(10, 5))
    plt.plot(range(1, N_EPOCHS + 1), train_losses, label='Train Loss')
    plt.plot(range(1, N_EPOCHS + 1), val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.legend()
    plt.title('Training and Validation Loss Over Time')
    plt.grid(True)
    loss_curve_path = os.path.join(RESULTS_FOLDER, 'train_val_loss_curve.png') 
    plt.savefig(loss_curve_path)
    plt.close()
    print(f"Train/Validation loss curve saved to {loss_curve_path}")


    print(f"\n--- FINAL TEST ---")
    all_test_X = []
    all_test_y = []
    test_files = 0
    test_file_list = [f for f in os.listdir(TEST_DATA_FOLDER) if f.lower().endswith('.csv')]

    if not test_file_list:
        print(f"Warning: No data")
    else:
        for filename in tqdm(test_file_list, desc=f"Loading Final Test CSVs from {TEST_DATA_FOLDER}"):
            file_path = os.path.join(TEST_DATA_FOLDER, filename)
            try:
                df = pd.read_csv(file_path)
                if not all(col in df.columns for col in IMU_FEATURES):
                    tqdm.write(f"Warning (Final Test): Skipping {filename}")
                    continue

                label = -1
                if 'stair' in filename.lower():
                    label = 1
                    df = df.iloc[:len(df) // 3] #only first third contains walking up
                elif 'treadmill' in filename.lower(): label = 0

                if label != -1:
                    
                    if len(df) > WINDOW_SIZE:
                        test_files += 1
                        file_X = df[IMU_FEATURES].values
                        file_y = np.full(len(file_X), label)
                        all_test_X.append(file_X)
                        all_test_y.append(file_y)
                    else:
                        tqdm.write(f"Warning Skipping {filename}")

            except pd.errors.EmptyDataError:
                tqdm.write(f"Warning Skipping empty file {filename}.")
            except Exception as e:
                tqdm.write(f"Error processing {filename}: {e}")

        if not all_test_X:
            print("Error: No valid final test data")
        else:
          
            X_test_concat = np.concatenate(all_test_X, axis=0)
            y_test_concat = np.concatenate(all_test_y, axis=0)
            del all_test_X, all_test_y
          
            try:
                scaler_X_loaded = joblib.load(SCALER_SAVE_PATH)
                X_test_scaled = scaler_X_loaded.transform(X_test_concat)
                del X_test_concat
        
                X_test_seq_final, y_test_seq_final = create_sequences(
                    X_test_scaled, y_test_concat, window_size=WINDOW_SIZE, step_size=50, desc="Creating Final Test sequences"
                )
                del X_test_scaled, y_test_concat

                if X_test_seq_final.size == 0:
                    print("Error: No final test sequences")
                else:

                    final_test_dataset = GaitActivityDataset(X_test_seq_final, y_test_seq_final)
                    del X_test_seq_final, y_test_seq_final
                    final_test_loader = DataLoader(final_test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
                  

                    if not os.path.exists(MODEL_SAVE_PATH):
                        print(f"Error: Model not found")
                    else:
                        model.load_state_dict(torch.load(MODEL_SAVE_PATH))
                        model.to(device)
                        model.eval()

                        all_preds_final = []
                        all_labels_final = []
                        final_eval_iterator = tqdm(final_test_loader, desc="Final Evaluation on Test Set", unit="batch")
                        with torch.no_grad():
                            for batch_X, batch_y in final_eval_iterator:
                                batch_X = batch_X.to(device, non_blocking=pin_memory)
                                outputs = model(batch_X)
                                _, predicted = torch.max(outputs.data, 1)
                                all_preds_final.extend(predicted.cpu().numpy())
                                all_labels_final.extend(batch_y.cpu().numpy()) 

                    
                        print("\n--- FINAL TEST SET METRICS ---")
                        accuracy = accuracy_score(all_labels_final, all_preds_final)
                        target_names = ['Level Ground', 'Stairs']
                        try:
                            report = classification_report(all_labels_final, all_preds_final, target_names=target_names, digits=4, zero_division=0)
                            conf_matrix = confusion_matrix(all_labels_final, all_preds_final)

                            print(f'Final Test Accuracy: {accuracy:.4f}')
                            print("\nFinal Test Classification Report:")
                            print(report)
                            print("\nFinal Test Confusion Matrix:")
                            print(conf_matrix)

                        except Exception as e:
                            print(f"\nError during final metric calculation")

            except FileNotFoundError:
                print(f"Error")
            except Exception as e:
                print(f"Error")
