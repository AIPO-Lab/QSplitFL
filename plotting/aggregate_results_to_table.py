import pandas as pd
import os
import glob
import math

# Configuration
DATASETS = ["MNIST", "FMNIST", "CIFAR10", "CIFAR-100"]
MODELS = ["CNN", "ResNet50", "MobileNetV4", "ConvNeXt"]
CLIENTS = [5, 10, 100, 200]
ROUNDS = [10, 20, 50, 100]

# Mapping dataset names to likely folder/file prefixes
DATASET_MAP = {
    "MNIST": {"folder": "MNIST_Results", "prefix": ["mnist"]},
    "FMNIST": {"folder": "FMNIST_Results", "prefix": ["fmnist"]},
    "CIFAR10": {"folder": "CIFAR10_Results", "prefix": ["cifar10"]},
    "CIFAR-100": {"folder": "CIFAR-100_Results", "prefix": ["cifar-100", "cifar100"]}
}

def get_result_file(dataset, model, client, round_num):
    info = DATASET_MAP[dataset]
    folder = info["folder"]
    prefixes = info["prefix"]
    
    # Try explicit model name first
    model_lower = model.lower()
    
    possible_filenames = []
    
    for prefix in prefixes:
        # 1. Explicit model name: e.g., mnist_resnet50_results...
        possible_filenames.append(f"{prefix}_{model_lower}_results_clients{client}_rounds{round_num}.csv")
        # 2. Results without model name (usually assumed to be CNN/Baseline)
        if model == "CNN":
            possible_filenames.append(f"{prefix}_results_clients{client}_rounds{round_num}.csv")

    for fname in possible_filenames:
        path = os.path.join(folder, fname)
        if os.path.exists(path):
            return path
            
    return None

def aggregate_data():
    results_list = []
    
    print(f"{'Dataset':<10} | {'Model':<12} | {'Clients':<8} | {'Rounds':<8} | {'Final Acc':<10} | {'Avg Split':<10}")
    print("-" * 80)
    
    for dataset in DATASETS:
        for model in MODELS:
            for client in CLIENTS:
                for round_num in ROUNDS:
                    file_path = get_result_file(dataset, model, client, round_num)
                    
                    if file_path:
                        try:
                            df = pd.read_csv(file_path)
                            if not df.empty and 'Accuracy' in df.columns and 'SplitLayer' in df.columns:
                                final_acc = df['Accuracy'].iloc[-1]
                                # Calculate average and apply ceiling
                                avg_split = math.ceil(df['SplitLayer'].mean())
                                
                                results_list.append({
                                    "Dataset": dataset,
                                    "Model": model,
                                    "Clients": client,
                                    "Rounds": round_num,
                                    "Final Accuracy": final_acc,
                                    "Avg Split Layer": avg_split
                                })
                                
                                print(f"{dataset:<10} | {model:<12} | {client:<8} | {round_num:<8} | {final_acc:.4f}     | {avg_split}")
                            else:
                                print(f"{dataset:<10} | {model:<12} | {client:<8} | {round_num:<8} | N/A (Empty)  | N/A")
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
                    else:
                        # Only print missing if strictly expecting it, helps reduce noise if running partials
                        # print(f"{dataset:<10} | {model:<12} | {client:<8} | {round_num:<8} | Not Found  | -")
                        pass

    # Create DataFrame
    results_df = pd.DataFrame(results_list)
    results_df.to_csv("final_comprehensive_summary.csv", index=False)
    print("\nSummary saved to 'final_comprehensive_summary.csv'")
    
    # Generate Markdown Table manually
    print("\n\n### Comprehensive Experiment Results Table")
    print("| Dataset | Model | Clients | Rounds | Final Accuracy | Avg Split Layer |")
    print("| :--- | :--- | :--- | :--- | :--- | :--- |")
    
    for item in results_list:
        print(f"| {item['Dataset']} | {item['Model']} | {item['Clients']} | {item['Rounds']} | {item['Final Accuracy']:.4f} | {item['Avg Split Layer']:.2f} |")

if __name__ == "__main__":
    aggregate_data()
