import pandas as pd
import os
import random
from PIL import Image

import torch
import torchvision
import torch.utils.data as data

cur_dir = os.path.dirname(os.path.abspath(__file__))


class Flickr30kDataset(data.Dataset):
    """
    Dataset loader for Flickr30k full datasets.
    """

    def __init__(self, root, transform = None):
        self.images_root = os.path.join(root, 'flickr30k_images')
        self.groups = pd.read_csv(os.path.join(root, 'results.csv'), sep='|').groupby(['image_name'])
        self.ids = [g[0] for g in self.groups]
        self.transform = transform or torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((224, 224)),
            torchvision.transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])
        self.images_length = len(self.ids)
        self.captions_length = self.images_length * 5

    def _get_image(self, index):
        filename = self.ids[index]
        image_file_path = os.path.join(self.images_root, filename)
        img = Image.open(image_file_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __getitem__(self, index):
        filename = self.ids[index]
        group_df = self.groups.get_group(filename)
        captions = group_df[' comment'].to_list()
        img = self._get_image(index)
        return img, captions

    def __len__(self):
        return self.images_length


class SiameseFlickr30kDataset(Flickr30kDataset):
    """
    Dataset loader for Flickr30k full datasets.
    """
    def __getitem__(self, index):
        image_id_index = int(index / 5)
        caption_id = int(index % 5)
        filename = self.ids[image_id_index]
        group_df = self.groups.get_group(filename)
        captions = group_df[' comment'].to_list()
        positive_img = self._get_image(image_id_index)
        return positive_img, captions[caption_id]

    def __len__(self):
        return self.captions_length


class TripletFlickr30kDataset(Flickr30kDataset):
    """
    Dataset loader for Flickr30k full datasets.
    """
    def __getitem__(self, index):
        image_id_index = int(index / 5)
        caption_id = int(index % 5)
        filename = self.ids[image_id_index]
        group_df = self.groups.get_group(filename)
        captions = group_df[' comment'].to_list()
        sample_negative_image_id = random.randint(0, self.images_length)
        positive_img = self._get_image(image_id_index)
        negative_img = self._get_image(sample_negative_image_id)
        return positive_img, negative_img, captions[caption_id]

    def __len__(self):
        return self.captions_length


def get_data_loader(root, batch_size=8, shuffle=False,
                    num_workers=1):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = SiameseFlickr30kDataset(root=root)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers)

    return data_loader
