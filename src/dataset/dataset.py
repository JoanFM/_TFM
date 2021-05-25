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

    def __getitem__(self, index):
        """This function returns a tuple that is further passed to collate_fn
        """
        filename = self.ids[index]
        group_df = self.groups.get_group(filename)
        captions = group_df[' comment'].to_list()

        image_file_path = os.path.join(self.images_root, filename)
        img = Image.open(image_file_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, img, captions[0]

    def __len__(self):
        return len(self.ids)


def get_data_loader(root, batch_size=8, shuffle=False,
                    num_workers=1):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""

    dataset = Flickr30kDataset(root=root)
    # Data loader
    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              pin_memory=True,
                                              num_workers=num_workers)

    return data_loader


def get_vocab():
    # 4154 words appear at least 10 times in the full 30k dataset
    import spacy
    from collections import Counter
    nlp = spacy.load('en_core_web_sm')
    dataset = Flickr30kDataset(root=os.path.join(cur_dir, '../../flickr30k_images'))
    vocab = set()
    vocab_count = Counter()
    c = []
    for (_, captions) in dataset:
        c.extend(captions)
    for i, doc in enumerate(nlp.pipe(c)):
        for token in doc:
            if not token.is_punct and not token.is_space:
                vocab.add(token.lemma_.lower())
                vocab_count[token.lemma_.lower()] += 1
        if i % 500 == 0:
            print(f' vocab size {len(vocab)} when processed {int(i / 5)} images')

    print(f' vocab size {len(vocab)}')

    import pickle
    with open('vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    import pickle
    with open('vocab_count.pkl', 'wb') as f:
        pickle.dump(vocab_count, f)


if __name__ == '__main__':
    get_vocab()
