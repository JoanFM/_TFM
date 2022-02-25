import os
import time
import torch

from src.model.vilt_model import ViltModel

DATASET_ROOT_PATH = os.getenv('DATASET_ROOT_PATH', '/hdd/master/tfm/flickr30k_images')
# The root path where the flickr30k entities per split is kept
DATASET_SPLIT_ROOT_PATH = os.getenv('DATASET_SPLIT_ROOT_PATH', '/hdd/master/tfm/flickr30k_images/flickr30k_entities')

VILT_BASE_MODEL_LOAD_PATH = os.getenv('VILT_BASE_MODEL_LOAD_PATH', '../vilt_irtr_f30k.ckpt')


def compute_recall():
    import copy
    from vilt import config
    from vilt.transforms.pixelbert import pixelbert_transform
    from src.dataset.dataset import get_image_data_loader, get_captions_data_loader
    from src.evaluate import evaluate

    # Scared config is immutable object, so you need to deepcopy it.
    conf = copy.deepcopy(config.config())
    conf['load_path'] = VILT_BASE_MODEL_LOAD_PATH
    conf['test_only'] = True
    conf['max_text_len'] = 40
    conf['max_text_len'] = 40
    conf['data_root'] = '/hdd/master/tfm/arrow'
    conf['datasets'] = ['f30k']
    conf['batch_size'] = 1
    conf['per_gpu_batchsize'] = 1
    conf['draw_false_image'] = 0
    conf['num_workers'] = 1

    # You need to properly configure loss_names to initialize heads (0.5 means it initializes head, but ignores the
    # loss during training)
    loss_names = {
        'itm': 0.5,
        'mlm': 0,
        'mpp': 0,
        'vqa': 0,
        'imgcls': 0,
        'nlvr2': 0,
        'irtr': 1,
        'arc': 0,
    }
    conf['loss_names'] = loss_names

    if torch.cuda.is_available():
        dev = 'cuda'
    else:
        dev = 'cpu'
    device = torch.device(dev)

    print(f' conf for ViltModel {conf}')

    vilt_model = ViltModel(conf)
    vilt_model.to(device)

    image_dataset = get_image_data_loader(root=DATASET_ROOT_PATH,
                                          split_root=DATASET_SPLIT_ROOT_PATH,
                                          split='test',
                                          transform=pixelbert_transform(384),
                                          batch_size=1)

    text_dataset = get_captions_data_loader(root=DATASET_ROOT_PATH,
                                            split_root=DATASET_SPLIT_ROOT_PATH,
                                            split='test',
                                            batch_size=1)

    images = []
    filenames = []
    for filenames_batch, images_batch in image_dataset:
        filenames.extend(filenames_batch)
        images.extend(images_batch)

    retrieved_image_filenames = []
    groundtruth_expected_image_filenames = []
    print(f' number of queries {len(text_dataset)}, against {len(images)}')
    for matching_filename, query in text_dataset:
        filename = matching_filename[0]
        groundtruth_expected_image_filenames.append([filename])
        q = query[0]
        start = time.time()
        scores = vilt_model.rank_query_vs_images(q, images)
        print(f' time to rank a single query {time.time() - start}s')
        # scores = vilt_model.rank(q, images)
        retrieved_image_filenames.append([f for _, f in sorted(zip(scores, filenames), reverse=True)])
        print(
            f' matching_filename {matching_filename[0]} in {retrieved_image_filenames[-1].index(matching_filename[0])}')

    evaluate(['recall', 'reciprocal_rank'], retrieved_image_filenames,
             groundtruth_expected_image_filenames,
             [1, 5, 10, 20, 100, 200, 500, None],
             {}, print_results=True)


def compute_recall_with_cache():
    import os
    from src.model.cached_scores import CachedScores
    from vilt.transforms.pixelbert import pixelbert_transform
    from src.dataset.dataset import get_image_data_loader, get_captions_data_loader
    from src.evaluate import evaluate

    image_dataset = get_image_data_loader(root=DATASET_ROOT_PATH,
                                          split_root=DATASET_SPLIT_ROOT_PATH,
                                          split='test',
                                          transform=pixelbert_transform(384),
                                          batch_size=1)

    text_dataset = get_captions_data_loader(root=DATASET_ROOT_PATH,
                                            split_root=DATASET_SPLIT_ROOT_PATH,
                                            split='test',
                                            batch_size=1)

    images = []
    filenames = []
    images_indices = []
    for indices, filenames_batch, images_batch in image_dataset:
        filenames.extend(filenames_batch)
        images.extend(images_batch)
        images_indices.extend(indices)

    retrieved_image_filenames = []
    groundtruth_expected_image_filenames = []
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    test_cache_scores = CachedScores(os.path.join(cur_dir, 'slow_scores/test.th'))
    for caption_indices, matching_filename, query in text_dataset:
        filename = matching_filename[0]
        groundtruth_expected_image_filenames.append([filename])
        scores = test_cache_scores.get_scores(caption_indices[0], images_indices)
        retrieved_image_filenames.append([f for _, f in sorted(zip(scores, filenames), reverse=True)])

    evaluate(['recall', 'reciprocal_rank'], retrieved_image_filenames,
             groundtruth_expected_image_filenames,
             [1, 5, 10, 20, 100, 200, 500, None],
             {}, print_results=True)


if __name__ == '__main__':
    compute_recall_with_cache()
