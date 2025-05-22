import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from sklearn.preprocessing import StandardScaler
from joblib import load
import glob
import os
import argparse
import numpy as np
from models import *

def load_model(model_path, device):
    model = torch.load(model_path, map_location=device)
    model.eval()
    return model

class CustomDataset(Dataset):
    def __init__(self, features, labels=None):
        self.features = features
        self.labels = labels
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.labels is not None:
            return torch.tensor(self.features[idx], dtype=torch.float32), torch.tensor(self.labels[idx], dtype=torch.float32)
        return torch.tensor(self.features[idx], dtype=torch.float32)

def main(args):
    # Load and preprocess the dataset
    df = pd.read_csv(args.dataset)
    print(f"Original dataset size: {len(df)}")
    df = df[df["Path_loss"] != float("inf")]
    print(f"Dataset size after removing inf Path_loss: {len(df)}")

    # Define input features
    input_columns=["Frequency_GHz"]
    if args.add_columns:
        add_columns_list = args.add_columns.split(",")
        if "GPS" in add_columns_list:
            GPS_columns=["Tx_x", "Tx_y", "Tx_z", "Rx_x", "Rx_y", "Rx_z"]
            input_columns=GPS_columns+input_columns
            add_columns_list.remove("GPS")

        if "Rx_z" in input_columns:
            # 找到 "Rx_z" 的索引位置
            index = input_columns.index("Rx_z")
            # 在 "Rx_z" 之後、"Frequency" 之前插入 add_columns_list
            input_columns = input_columns[:index+1] + add_columns_list + input_columns[index+1:]
        else:
            input_columns = add_columns_list + input_columns

    x_test = df[input_columns]
    y_test = df['Path_loss']

    # Load scaler and scale input data
    # print(f"model folder: {glob.glob(f'{args.model_folder}/scaler*.joblib')}")
    scaler = load(glob.glob(f'{args.model_folder}/scaler*.joblib')[0])
    x_test_scaled = scaler.transform(x_test)

    # Prepare the dataset and dataloader
    test_dataset = CustomDataset(x_test_scaled, y_test.to_numpy())
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Set up the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model_path = glob.glob(f'{args.model_folder}/best_model*.pth')[0]
    model = load_model(model_path, device)
    print(f"Loaded model: {model_path}")

    # Model information
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total model parameters: {total_params}")

    # Evaluation
    predicted_list = []
    true_list = []
    model.eval()
    # 推論時間計算
    total_time = 0.0
    num_batches = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            # 確保 GPU 操作完成
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            # 使用 time.perf_counter() 開始計時
            start_time = time.perf_counter()
            outputs = model(inputs)
            # 同樣在推論後進行同步操作（如果使用 GPU）
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.perf_counter()

            # 計算並累加當前 batch 的推論時間
            batch_time = end_time - start_time
            total_time += batch_time
            num_batches += 1

            predicted_list.extend(outputs.squeeze().cpu().numpy())
            true_list.extend(labels.cpu().numpy())

    # 計算平均每個 batch 的推論時間
    avg_time_per_batch = total_time / num_batches
    print(f"Average inference time per batch: {avg_time_per_batch:.9f} seconds")
    # Calculate metrics
    mse = np.mean((np.array(true_list) - np.array(predicted_list))**2)
    mae = np.mean(np.abs(np.array(true_list) - np.array(predicted_list)))
    r2 = 1 - (np.sum((np.array(true_list) - np.array(predicted_list))**2) / np.sum((np.array(true_list) - np.mean(np.array(true_list)))**2))

    print(f"Mean Squared Error: {mse}")
    print(f"Mean Absolute Error: {mae}")
    print(f"R-squared: {r2}")

    # Save results
    if args.save_csv:
        df['predicted_path_loss'] = pd.Series(predicted_list)
        if not os.path.exists(args.save_path):
            os.makedirs(args.save_path)
        if args.model_name=="Transformer":
            if "layer2" in args.model_folder:
                args.model_name=args.model_name+"2layer"
            if "layer4" in args.model_folder:
                args.model_name=args.model_name+"4layer"
            if "layer6" in args.model_folder:
                args.model_name=args.model_name+"6layer"
            if "layer8" in args.model_folder:
                args.model_name=args.model_name+"8layer"

        output_file = f"{args.save_path}/{args.model_name}_{args.tag}_results.csv"
        df.to_csv(output_file, index=False)
        print(f"Results saved to: {output_file}")

if __name__ == "__main__":
    print("----------------------------------")
    parser = argparse.ArgumentParser(description="Test a trained model and save results to CSV")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the test dataset CSV file")
    parser.add_argument("--model_folder", type=str, required=True, help="Path to the folder containing the trained model and scaler")
    parser.add_argument("--model_name", type=str, required=True, help="Model name (example: \"MLP\", \"GeoSigNet\")")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size for testing")
    parser.add_argument("--add_columns", type=str, default=None, help="Additional columns to include in input features")
    parser.add_argument("--tag", type=str, default=None, help="Tag info")
    parser.add_argument("--save_path", type=str, default="temp_result", help="Save to selected folder (last character do not be /)")
    parser.add_argument("--save_csv", action="store_true",  help="save to csv file")
    args = parser.parse_args()

    main(args)
    print("----------------------------------")