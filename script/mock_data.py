"""
@author: bfx
@version: 1.0.0
@file: mock_data.py
@time: 1/12/25 13:29
"""
import os

import pandas as pd

def mock_data(label_file, img_folder, out_path):
    """
    Mock data for training.

    Args:
        label_file (str): Path to the CSV file containing labels.
        img_folder (str): Path to the folder containing images.
        out_path (str): Path to save the output CSV file (img_path, event_time, label).

    Returns:
        None
    """
    df_labels = pd.read_csv(label_file)
    img_paths = os.listdir(img_folder)
    img_paths = img_paths[:len(df_labels)]
    img_paths = [os.path.join(img_folder, img_path) for img_path in img_paths]
    df_labels['img_path'] = img_paths
    df_labels.to_csv(out_path, index=False)
    print(f"Mock data saved to {out_path}. {len(df_labels)} samples created.")
    print(df_labels.head())


def delete_files_starting_with_dot_underscore(directory):
    """
    Deletes all files starting with ._ in the specified directory.

    Parameters:
    directory (str): The directory path to search for files.
    """
    try:
        # List all files in the specified directory
        for filename in os.listdir(directory):
            if filename.startswith('._'):
                file_path = os.path.join(directory, filename)
                os.remove(file_path)
                print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Error: {e}")



if __name__ == '__main__':


    label_file = '../sample data/METABRIC/label.csv'
    img_folder = 'C:/Users/fbu/PycharmProjects/DeepHit-PyTorch/sample data/img_data/'
    out_path = '../sample data/img_data/mock_training_data.csv'

    delete_files_starting_with_dot_underscore(img_folder)

    mock_data(label_file, img_folder, out_path)




