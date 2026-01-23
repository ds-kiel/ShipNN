import os
import yaml
import pandas as pd

def list_files_with_split_type(root_dir):
    file_data = []
    # Traverse through the root directory
    for split_type in ['train', 'test', 'validation']:
        split_path = os.path.join(root_dir, split_type, 'audio')
        if os.path.exists(split_path):
            for root, _, files in os.walk(split_path):
                for file in files:
                    file_path = str(os.path.join(root, file))
                    file_index = int(os.path.splitext(file)[0])
                    ship_type = root.split('/')[-1]
                    file_data.append({'file_path': file_path, 
                                    'split_type': split_type, 
                                    'file_index': file_index, 
                                    'ship_type': ship_type})

    df = pd.DataFrame(file_data)
    return df

def get_combined_metadata(root_dir):
    m1 = pd.read_csv(root_dir+"/train/metadata.csv")
    m2 = pd.read_csv(root_dir+"/test/metadata.csv")
    m3 = pd.read_csv(root_dir+"/validation/metadata.csv")
    m1['split_type'] = 'train'
    m2['split_type'] = 'test'
    m3['split_type'] = 'validation'
    metadatas = pd.concat([m1,m2,m3], axis=0)
    return metadatas

def create_correct_metadata(data_dir):
    for root_dir in ["inclusion_2000_exclusion_4000","inclusion_3000_exclusion_5000","inclusion_4000_exclusion_6000"]:
        df = list_files_with_split_type(data_dir+root_dir)
        metadata = get_combined_metadata(data_dir+root_dir)
        df = df.merge(metadata[['split_type', 'file_index', 'MMSI']], 
                    on=['split_type', 'file_index'], 
                    how='left')
        df.to_csv(data_dir+root_dir+"/ShipNN_metadata.csv")


if __name__ == "__main__":
    # Load configuration
    with open("../config/mobilenet.yaml", "r") as file:
        config = yaml.safe_load(file)

    create_correct_metadata(config["dataset"]["root_dir"])