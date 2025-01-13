"""
@author: bfx
@version: 1.0.0
@file: import_data_with_images.py.py
@time: 1/12/25 13:26
"""
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from import_data import f_get_Normalization, f_get_fc_mask2, f_get_fc_mask3
from models.resnet50 import ResNet50
from torchvision import transforms
from PIL import Image
import torch
from tqdm import tqdm
tqdm.pandas()  # Enable the progress bar for pandas apply

# Custom dataset for batch processing
class ImageDataset(Dataset):
    def __init__(self, img_paths, transform):
        self.img_paths = img_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert('RGB')
        img = self.transform(img)
        return img

def extract_image_features_batch(img_paths, batch_size=64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img_extractor = ResNet50(pretrained=True).to(device)
    img_extractor.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    dataset = ImageDataset(img_paths, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    features_list = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting image features in batches"):
            batch = batch.to(device)
            features = img_extractor(batch)
            features_list.append(features.cpu().numpy())

    return np.vstack(features_list)

def import_dataset_image(norm_mode='standard'):
    in_filename = './sample data/img_data/mock_training_data.csv'
    df = pd.read_csv(in_filename, sep=',')

    # Extract image features in batches and replace the previous data columns
    print("Extracting image features...")
    img_paths = df['img_path'].tolist()
    image_features = extract_image_features_batch(img_paths, batch_size=32)

    # Convert image features to a proper numpy array format
    data = np.array(image_features)

    # Save the extracted features to a CSV file
    final_data = pd.concat([df[['event_time', 'label']], pd.DataFrame(data)], axis=1)
    final_data.columns = ['event_time', 'label'] + [f'feature_{i}' for i in range(data.shape[1])]
    final_data.to_csv('./sample data/img_data/mock_training_data_features.csv', index=False)

    # Prepare the data for training
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

    return DIM, DATA, MASK