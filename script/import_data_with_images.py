"""
@author: bfx
@version: 1.0.0
@file: import_data_with_images.py.py
@time: 1/12/25 13:26
"""
import numpy as np
import pandas as pd

from import_data import f_get_Normalization, f_get_fc_mask2, f_get_fc_mask3
from models.resnet50 import ResNet50
from torchvision import transforms
from PIL import Image
import torch

# 初始化模型


def extract_image_features(img_path):
    img_extractor = ResNet50(pretrained=True)
    img_extractor.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    img = Image.open(img_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = img_extractor(img_tensor)
    return features.numpy()

def import_dataset_image(norm_mode='standard'):
    in_filename = '../sample data/img_data/mock_training_data.csv'
    df = pd.read_csv(in_filename, sep=',')

    # Extract image features and replace the previous data columns
    df['image_features'] = df['img_path'].apply(extract_image_features)

    # Convert image features to a proper numpy array format
    data = np.vstack(df['image_features'].values)

    label = np.asarray(df[['label']])
    time = np.asarray(df[['event_time']])
    data = f_get_Normalization(data, norm_mode)

    num_Category = int(np.max(time) * 1.2)
    num_Event = int(len(np.unique(label)) - 1)

    x_dim = data.shape[1]

    mask1 = f_get_fc_mask2(time, label, num_Event, num_Category)
    mask2 = f_get_fc_mask3(time, -1, num_Category)

    DIM = x_dim
    DATA = (data, time, label)
    MASK = (mask1, mask2)

    return DIM, DATA, MASK