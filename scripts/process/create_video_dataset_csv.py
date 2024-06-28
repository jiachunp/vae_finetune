import os
import pandas as pd


def list_files_in_folder(folder_path):
    # List all mp4 files in the folder and its subfolders
    file_paths = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.mp4'):  # Check if the file is a .mp4 file
                file_path = os.path.join(root, file)
                if os.path.getsize(file_path) > 0:  # Exclude files with size 0
                    file_paths.append(file_path)
    
    # Create a DataFrame
    df = pd.DataFrame(file_paths, columns=['videoid'])
    
    # Save to CSV
    csv_path = "/mnt/data/public/dataset/talking_head/video_datasets/metadata.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV file saved to: {csv_path}")

    # Output the number of rows in the CSV file
    num_rows = df.shape[0]
    print(f"Number of rows in CSV file: {num_rows}")


# Example usage
folder_path = '/mnt/data/public/dataset/talking_head/video_datasets'
list_files_in_folder(folder_path)
