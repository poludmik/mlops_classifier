import torch
import torch.nn as nn
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# Module doesn't work

def load_images(image_path):
    if image_path.endswith('.npy'):
        images = np.load(image_path)
        images = torch.from_numpy(images).float()
    else:
        print('Image format not supported (please use .npy)')
    return images

def extract_features(model, dataloader):
    model.eval()
    features = []
    for images in dataloader:
        with torch.no_grad():
            out = torch.cat([model(batch) for batch in dataloader], 0)
            features.append(out)
    return torch.cat(features, dim=0)

def viz_main():
    model_path = 'models/MyAwesomeModel/model.pt'
    data_path = 'data/raw/random_mnist_images/mnist_sample.npy'
    
    # Load the pre-trained model
    model = torch.load(model_path)
    model = nn.Sequential(*list(model.children())[:-1])  # Remove the last layer

    # Load the data
    images = load_images(data_path)[0].unsqueeze(0)
    dataloader = torch.utils.data.DataLoader(images, batch_size=64)

    # Extract features
    features = extract_features(model, dataloader)[0]

    print(features.shape)

    # Perform t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    features_2d = tsne.fit_transform(features)

    # Plot and save the visualization
    plt.scatter(features_2d[:, 0], features_2d[:, 1])
    os.makedirs('reports/figures', exist_ok=True)
    plt.savefig('reports/figures/tsne_visualization.png')
    plt.show()