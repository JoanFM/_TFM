import pandas as pd
import os
from PIL import Image

import torch
import torchvision
import torch.utils.data as data

import torchvision.datasets as dset

cur_dir = os.path.dirname(os.path.abspath(__file__))

DEFAULT_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.CenterCrop((224, 224)),
    torchvision.transforms.RandomHorizontalFlip(p=0.5),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])


DEFAULT_TRAIN_TRANSFORM = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    torchvision.transforms.RandomResizedCrop(size=(224, 224), scale=(0.08, 1.0), ratio=(3. / 4., 4. / 3.)),
    torchvision.transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )])


class OnlyCaptionsCocoCaptions(dset.CocoCaptions):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, index: int):
        id = self.ids[index]
        image = None
        target = self._load_target(id)

        return image, target


class COCODataset(data.Dataset):
    """
    Dataset loader for COCO full datasets.
    """

    def __init__(self, root, split_root, split, transform=None, **kwargs):
        image_split = split
        ann_split = split
        if split == 'validation':
            ann_split = 'val'
        elif split == 'val':
            image_split = 'validation'

        self.images_root = os.path.join(root, f'{image_split}/data')
        self.annotations_file = os.path.join(root, f'raw/captions_{ann_split}2014.json')
        self.transform = transform
        self.inner_dataset = dset.CocoCaptions(root=self.images_root, annFile=self.annotations_file,
                                               transform=self.transform)

    def _get_matching_filename(self, index):
        return self.inner_dataset.coco.loadImgs(self.inner_dataset.ids[index])[0]["file_name"]

    def _get_matching_filename_from_caption_idx(self, index):
        return self.inner_dataset.coco.loadImgs(self.inner_dataset.ids[index // 5])[0]["file_name"]

    def _get_image(self, index):
        image, _ = self.inner_dataset[index]
        return image

    def _get_caption(self, index):
        image_index = index // 5
        caption_inner_index = index % 5
        _, captions = self.inner_dataset[image_index]
        return captions[caption_inner_index]


class ImageCOCODataset(COCODataset):
    """
    Dataset loader for COCO full datasets.
    """

    def __getitem__(self, index):
        return self._get_image(index)

    def __len__(self):
        return len(self.inner_dataset)


class CaptionCOCODataset(COCODataset):
    """
    Dataset loader for COCO full datasets.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inner_dataset = OnlyCaptionsCocoCaptions(root=self.images_root, annFile=self.annotations_file,
                                                      transform=self.transform)

    def __getitem__(self, index):
        return index, self._get_matching_filename(index // 5), self._get_caption(index)

    def __len__(self):
        return len(self.inner_dataset) * 5


class ImageCaptionCOCODataset(COCODataset):
    """
    Dataset loader for COCO full datasets.
    """

    def __getitem__(self, index):
        caption_index = index
        image_index = index // 5
        caption_inner_index = index % 5
        positive_img, captions = self.inner_dataset[image_index]
        caption = captions[caption_inner_index]
        matching_filename = self._get_matching_filename(image_index)
        return caption_index, image_index, matching_filename, positive_img, caption

    def __len__(self):
        return len(self.inner_dataset) * 5


class Flickr30kDataset(data.Dataset):
    """
    Dataset loader for Flickr30k full datasets.
    """

    def __init__(self, root, split_root, split, transform=None, **kwargs):
        self.images_root = os.path.join(root, 'flickr30k-images')
        with open(os.path.join(split_root, f'{split}.txt'), 'r') as f:
            self.ids = f.read().split('\n')[0: -1]
        self.groups = pd.read_csv(os.path.join(root, 'results.csv'), sep='|').groupby(['image_name'])
        self.transform = transform or DEFAULT_TRANSFORM
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
        return index, filename, img

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
        return index, self._get_matching_filename(index), self.captions[index]

    def __len__(self):
        return self.captions_length

    def _get_matching_filename(self, index):
        return self.matching_filenames[index]

    _get_matching_filename_from_caption_idx = _get_matching_filename


class ImageCaptionFlickr30kDataset(Flickr30kDataset):
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
        self.transform = None

    def _get_image(self, filename):
        image_file_path = os.path.join(self.images_root, filename)
        img = Image.open(image_file_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img

    def _get_matching_filename(self, index):
        return self.matching_filenames[index]

    def __getitem__(self, index):
        caption_index = index
        image_index = index // 5
        matching_filename = self._get_matching_filename(index)
        caption = self.captions[caption_index]
        positive_img = self._get_image(matching_filename)
        return caption_index, image_index, matching_filename, positive_img, caption

    def __len__(self):
        return self.captions_length

    @property
    def num_images(self):
        return len(self.filenames)

    _get_matching_filename_from_caption_idx = _get_matching_filename


class ImageConcatenatedCaptionsFlickr30kDataset(Flickr30kDataset):
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
        return 128  # self.images_length


from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler


class SampleBatchNoCommonImmages(BatchSampler):
    """
    Class guarantees that in the same batch not 2 captions with the same corresponding image appear in the same batch
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        from collections import defaultdict
        idx_by_matching_filename = defaultdict(list)
        for idx in self.sampler:
            corresponding_positive_image_filename = self.sampler.data_source._get_matching_filename_from_caption_idx(
                idx)
            idx_by_matching_filename[corresponding_positive_image_filename].append(idx)
            if len(idx_by_matching_filename.keys()) == self.batch_size:
                batch_to_return = []
                keys = list(idx_by_matching_filename.keys())
                for key in keys:
                    batch_to_return.append(idx_by_matching_filename[key].pop())
                    if len(idx_by_matching_filename[key]) == 0:
                        del idx_by_matching_filename[key]
                yield batch_to_return

        # Here all the `idx` of the sampler have been put in the dictionary
        if not self.drop_last:
            end = False
            while not end:
                batch_to_return = []
                keys = list(idx_by_matching_filename.keys())
                for key in keys:
                    batch_to_return.append(idx_by_matching_filename[key].pop())
                    if len(idx_by_matching_filename[key]) == 0:
                        del idx_by_matching_filename[key]

                end = (len(idx_by_matching_filename.keys()) == 0)
                if len(batch_to_return) > 0:
                    yield batch_to_return


def get_concatenated_captions_image_data_loader(root, split_root, split, batch_size=8, shuffle=False,
                                                num_workers=1, batch_sampling=False, **kwargs):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = ImageConcatenatedCaptionsFlickr30kDataset(root=root, split_root=split_root, split=split, **kwargs)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=kwargs['collate_fn'] if 'collate_fn' in kwargs else None,
                                              batch_sampler=kwargs[
                                                  'batch_sampler'] if 'batch_sampler' in kwargs else None)

    return data_loader


def get_captions_image_data_loader(root, split_root, split, batch_size=8, shuffle=False,
                                   num_workers=1, batch_sampling=False, dset='flickr', **kwargs):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = ImageCaptionFlickr30kDataset(root=root, split_root=split_root, split=split,
                                           **kwargs) if dset == 'flickr' else ImageCaptionCOCODataset(root=root,
                                                                                                      split_root=None,
                                                                                                      split=split,
                                                                                                      **kwargs)

    batch_sampler = None

    if batch_sampling:
        if shuffle:
            sampler = RandomSampler(dataset, generator=None)
        else:
            sampler = SequentialSampler(dataset)

        batch_sampler = SampleBatchNoCommonImmages(sampler, batch_size, False)
        # Data loader
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  pin_memory=True,
                                                  num_workers=num_workers,
                                                  collate_fn=kwargs['collate_fn'] if 'collate_fn' in kwargs else None,
                                                  batch_sampler=batch_sampler)
    else:
        data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  num_workers=num_workers,
                                                  collate_fn=kwargs['collate_fn'] if 'collate_fn' in kwargs else None)

    return data_loader


def get_image_data_loader(root, split_root, split, batch_size=8, shuffle=False,
                          num_workers=1, batch_sampling=False, dset='flickr', **kwargs):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = ImageFlickr30kDataset(root=root, split_root=split_root, split=split,
                                    **kwargs) if dset == 'flickr' else ImageCOCODataset(root=root, split_root=None,
                                                                                        split=split, **kwargs)

    if 'force_transform_to_none' in kwargs:
        dataset.transform = None
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=kwargs['collate_fn'] if 'collate_fn' in kwargs else None,
                                              batch_sampler=kwargs[
                                                  'batch_sampler'] if 'batch_sampler' in kwargs else None)

    return data_loader


def get_captions_data_loader(root, split_root, split, batch_size=8, shuffle=False,
                             num_workers=1, batch_sampling=False, dset='flickr', **kwargs):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = CaptionFlickr30kDataset(root=root, split_root=split_root, split=split,
                                      **kwargs) if dset == 'flickr' else CaptionCOCODataset(root=root, split_root=None,
                                                                                            split=split, **kwargs)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers,
                                              collate_fn=kwargs['collate_fn'] if 'collate_fn' in kwargs else None,
                                              batch_sampler=kwargs[
                                                  'batch_sampler'] if 'batch_sampler' in kwargs else None)

    return data_loader
