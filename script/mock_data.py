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
    df_labels['img_path'] = img_paths
    df_labels.to_csv(out_path, index=False)
    print(f"Mock data saved to {out_path}. {len(df_labels)} samples created.")
    print(df_labels.head())

if __name__ == '__main__':
    label_file = '../sample data/METABRIC/label.csv'
    img_folder = '../sample data/img_data/'
    out_path = '../sample data/img_data/mock_training_data.csv'
    mock_data(label_file, img_folder, out_path)




