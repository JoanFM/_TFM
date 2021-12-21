import pandas as pd
import os
from PIL import Image

import torch
import torchvision
import torch.utils.data as data

cur_dir = os.path.dirname(os.path.abspath(__file__))


class Flickr30kDataset(data.Dataset):
    """
    Dataset loader for Flickr30k full datasets.
    """

    def __init__(self, root, split_root, split, transform=None):
        self.images_root = os.path.join(root, 'flickr30k-images')
        with open(os.path.join(split_root, f'{split}.txt'), 'r') as f:
            self.ids = f.read().split('\n')[0: -1]
        self.groups = pd.read_csv(os.path.join(root, 'results.csv'), sep='|').groupby(['image_name'])
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
        filename = f'{self.ids[index]}.jpg'
        image_file_path = os.path.join(self.images_root, filename)
        img = Image.open(image_file_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img


class ImageFlickr30kDataset(Flickr30kDataset):
    """
    Dataset loader for Flickr30k full datasets.
    """

    def __getitem__(self, index):
        filename = f'{self.ids[index]}.jpg'
        img = self._get_image(index)
        return filename, img

    def __len__(self):
        return self.images_length


class CaptionFlickr30kDataset(Flickr30kDataset):
    """
    Dataset loader for Flickr30k full datasets.
    """
    def __init__(self, root, *args, **kwargs):
        super().__init__(root=root, *args, **kwargs)
        self.df = pd.read_csv(os.path.join(root, 'results.csv'), sep='|')
        self.filenames = [f'{i}.jpg' for i in self.ids]
        self.df = self.df[self.df['image_name'].isin(self.filenames)]
        self.captions = self.df[' comment'].values
        self.matching_filenames = self.df['image_name'].values

    def __getitem__(self, index):
        return self.matching_filenames[index], self.captions[index]

    def __len__(self):
        return self.captions_length


class ImageCaptionFlickr30kDataset(Flickr30kDataset):
    """
    Dataset loader for Flickr30k full datasets.
    """

    def __getitem__(self, index):
        image_id_index = int(index)
        filename = f'{self.ids[image_id_index]}.jpg'
        group_df = self.groups.get_group(filename)
        captions = group_df[' comment'].to_list()
        captions_concat = ' '.join(captions)
        positive_img = self._get_image(image_id_index)
        return filename, positive_img, captions_concat

    def __len__(self):
        return self.images_length


def get_data_loader(root, split_root, split, batch_size=8, shuffle=False,
                    num_workers=1):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = ImageCaptionFlickr30kDataset(root=root, split_root=split_root, split=split)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers)

    return data_loader


def get_image_data_loader(root, split_root, split, batch_size=8, shuffle=False,
                          num_workers=1):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = ImageFlickr30kDataset(root=root, split_root=split_root, split=split)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers)

    return data_loader


def get_captions_data_loader(root, split_root, split, batch_size=8, shuffle=False,
                             num_workers=1):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = CaptionFlickr30kDataset(root=root, split_root=split_root, split=split)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers)

    return data_loader
