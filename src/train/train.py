import torch
import time
import os
import pickle
from collections import deque

from src.model import ImageEncoder
from src.model import TextEncoder
from src.dataset import get_data_loader

cur_dir = os.path.dirname(os.path.abspath(__file__))

l1_regularization_weight = 1e-7


def train(output_model_path: str = os.path.join(cur_dir, 'output_models'),
          vectorizer_path: str = os.path.join(cur_dir, '../../vectorizer.pkl'),
          num_epochs: int = 3):
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)

    os.makedirs(output_model_path, exist_ok=True)
    image_encoder = ImageEncoder()
    text_encoder = TextEncoder(vectorizer_path)
    optimizer = torch.optim.Adam(image_encoder.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss_fn = torch.nn.CosineEmbeddingLoss(margin=0)

    for epoch in range(num_epochs):
        train_data_loader = get_data_loader(root=os.path.join(cur_dir, '../../flickr30k_images'), shuffle=True,
                                            batch_size=16)
        train_loss = []
        loss_val = 0
        time_start = time.time()
        queue = deque(maxlen=20)
        positive_tensor = torch.Tensor([1]).to(device)

        for batch_id, (image, caption) in enumerate(train_data_loader, 1):
            image_encoder.feature_extractor.model.eval()
            image_encoder.sparse_encoder.train()
            optimizer.zero_grad()
            image = image.to(device)
            image_embedding = image_encoder.forward(image)
            text_embedding = text_encoder.forward(caption)
            text_embedding = text_embedding.to(device)
            l1_regularization = torch.mean(torch.sum(image_embedding, dim=1))
            loss = loss_fn(image_embedding, text_embedding,
                           positive_tensor) + l1_regularization_weight * l1_regularization
            loss_val += loss.item()
            loss.backward()
            optimizer.step()
            if batch_id % 100 == 0:
                print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
                    batch_id, loss_val / 10, time.time() - time_start))
                loss_val = 0
                time_start = time.time()
            if batch_id % 1000 == 0:
                torch.save(image_encoder.state_dict(),
                           output_model_path + '/model-inter-' + str(epoch + 1) + '-' + str(batch_id) + ".pt")
            if batch_id % 1000 == 0:
                image_encoder.feature_extractor.model.eval()
                image_encoder.sparse_encoder.eval()
                # TODO: Test some batches
                right, error = 0, 1
                queue.append(right * 1.0 / (right + error))
            train_loss.append(loss_val)

        with open(f'train_loss-{epoch}', 'wb') as f:
            pickle.dump(train_loss, f)

        acc = 0.0
        for d in queue:
            acc += d
        print("#" * 70)
        print(f"accuracy at the end of epoch {epoch}/{num_epochs}: ", acc / 20)


if __name__ == '__main__':
    train()
