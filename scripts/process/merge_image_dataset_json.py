import json

# Define the paths to the JSON files
file1_path = '/mnt/data/public/dataset/talking_head/image_datasets/HDTF/metadata.json'
file2_path = '/mnt/data/public/dataset/talking_head/image_datasets/VFHQ/metadata.json'
output_file_path = '/mnt/data/public/dataset/talking_head/image_datasets/metadata.json'

# Load the data from the first JSON file
with open(file1_path, 'r') as file1:
    data1 = json.load(file1)

# Load the data from the second JSON file
with open(file2_path, 'r') as file2:
    data2 = json.load(file2)

# Merge the two dictionaries
# # If both JSON files have dictionaries at the top level, use this
# merged_data = {**data1, **data2}

# If both JSON files have lists at the top level, use this
merged_data = data1 + data2

# Save the merged data to a new JSON file
with open(output_file_path, 'w') as output_file:
    json.dump(merged_data, output_file, indent=2)

print(f"Merged JSON saved to {output_file_path}")
