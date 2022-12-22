import torch
from torch.utils.data import DataLoader

from nnet.loss import AlphaLoss
from nnet.dataset import BoardData
from settings import BATCH_SIZE


def train(nnet, lr, dataset, epoch_start=0, epoch_end=20):
    """
    Inputs:
        - nnet: neural net to train
        - lr: learning rate. This is controlled in settings.py rather than using a scheduler since it changes accross multiple 
        training sets.
        - dataset: data used to train nnet
    """
    # check if cuda is available
    cuda = torch.cuda.is_available()
    old_params = nnet.parameters()
    optimizer = torch.optim.SGD(old_params, lr=lr, momentum=0.9)
    criterion = AlphaLoss()
    train_set = BoardData(dataset=dataset)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    for epoch in range(epoch_start, epoch_end):
        total_loss = 0.0
        losses_per_batch = []
        for idx, data in enumerate(train_loader):
            s, p, v = data
            if cuda:
                s, p, v = s.cuda(), p.cuda(), v.cuda()
            optimizer.zero_grad()
            p_pred, v_pred = nnet(s)
            loss = criterion(p_pred, p, v_pred, v)
            loss.backward()
            total_loss += loss.item()
