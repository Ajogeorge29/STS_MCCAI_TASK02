import argparse
import os
import torch
from torch.utils.data import DataLoader
from organised_complete_validation_file import PointNetLKUnifiedModel, UnlabeledPointCloudDataset, generate_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference using PointNetLK model")
    parser.add_argument("--model_weights", type=str, required=True, help="Path to model .pth file")
    parser.add_argument("--input_dir", type=str, default="/inputs", help="Input directory for validation .npy files")
    parser.add_argument("--output_dir", type=str, default="/outputs", help="Directory to save output transformation matrices")
    return parser.parse_args()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("📦 Loading model...")
    model = PointNetLKUnifiedModel(num_iterations=10)
    model.load_state_dict(torch.load(args.model_weights, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))

    validation_ds = UnlabeledPointCloudDataset(root_dir=args.input_dir, n_points=2048)
    validation_loader = DataLoader(validation_ds, batch_size=8, shuffle=False)

    print("🔍 Running inference...")
    generate_predictions(model, validation_loader, args.output_dir)

if __name__ == "__main__":
    main()
