import torch
import time
import os
import pickle
from collections import deque

from src.model import ImageSiamese
from src.model import TextEncoder
from src.dataset import get_data_loader

cur_dir = os.path.dirname(os.path.abspath(__file__))


def train(output_model_path: str = os.path.join(cur_dir, 'output_models'), vectorizer_path: str = os.path.join(cur_dir, '../../vectorizer.pkl')):
    os.makedirs(output_model_path, exist_ok=True)
    siamese_net = ImageSiamese()
    text_encoder = TextEncoder(vectorizer_path)
    optimizer = torch.optim.Adam(siamese_net.parameters(), lr=0.01)
    optimizer.zero_grad()
    loss_fn = torch.nn.TripletMarginWithDistanceLoss(distance_function=torch.nn.CosineSimilarity())

    train_data_loader = get_data_loader(root=os.path.join(cur_dir, '../../flickr30k_images'), shuffle=True)

    train_loss = []
    loss_val = 0
    time_start = time.time()
    queue = deque(maxlen=20)

    for batch_id, (image_pos, image_neg, caption) in enumerate(train_data_loader, 1):
        siamese_net.feature_extractor.model.eval()
        siamese_net.sparse_encoder.train()
        optimizer.zero_grad()
        positive_output, negative_output = siamese_net.forward(image_pos, image_neg)
        text_output = text_encoder.forward(caption)
        print(f' positive_output size {positive_output.size()}')
        print(f' negative_output size {negative_output.size()}')
        print(f' text_output size {text_output.size()}')
        positive_dist = torch.functional.F.cosine_similarity(text_output, positive_output, 1, 1e-18)
        negative_dist = torch.functional.F.cosine_similarity(text_output, negative_output, 1, 1e-18)
        print(f' positive_dist {positive_dist}')
        print(f' negative_dist {negative_dist}')
        loss = loss_fn(text_output, positive_output, negative_output)
        print(f' loss {loss}')

        loss_val += loss.item()
        loss.backward()
        optimizer.step()
        if batch_id % 10 == 0:
            print('[%d]\tloss:\t%.5f\ttime lapsed:\t%.2f s' % (
                batch_id, loss_val / 10, time.time() - time_start))
            loss_val = 0
            time_start = time.time()
        if batch_id % 20 == 0:
            torch.save(siamese_net.state_dict(), output_model_path + '/model-inter-' + str(batch_id + 1) + ".pt")
        if batch_id % 30 == 0:
            siamese_net.feature_extractor.model.eval()
            siamese_net.sparse_encoder.eval()
            # TODO: Test some batches
            right, error = 0, 1
            queue.append(right * 1.0 / (right + error))
        train_loss.append(loss_val)

    with open('train_loss', 'wb') as f:
        pickle.dump(train_loss, f)

    acc = 0.0
    for d in queue:
        acc += d
    print("#" * 70)
    print("final accuracy: ", acc / 20)


if __name__ == '__main__':
    train()
