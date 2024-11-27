import argparse
from pathlib import Path
import pandas as pd


def list_files_in_folder(folder_path, output_path):
    # List all mp4 files in the folder and its subfolders
    file_paths = [str(file) for file in Path(folder_path).rglob('*.mp4')]
    
    # Create a DataFrame
    df = pd.DataFrame(file_paths, columns=['video'])
    
    # Save to CSV
    df[64:].to_csv(output_path, index=False)
    print(f"Train CSV file saved to: {output_path}")

    # Output the number of rows in the CSV file
    print(f"Number of rows in CSV file: {df.shape[0]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder_path", type=str, help="Path to the folder containing mp4 files")
    parser.add_argument("--output_path", type=str, help="Path to save the CSV file")
    args = parser.parse_args()

    list_files_in_folder(args.folder_path, args.output_path)
