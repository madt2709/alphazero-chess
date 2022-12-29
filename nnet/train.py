import torch
import matplotlib.pyplot as plt
import datetime
import os
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
    avg_loss_per_epoch = []
    optimizer = torch.optim.SGD(old_params, lr=lr, momentum=0.9)
    criterion = AlphaLoss()
    train_set = BoardData(dataset=dataset)
    train_loader = DataLoader(
        train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    for epoch in range(epoch_start, epoch_end):
        total_loss = 0.0
        losses_per_batch = []  # loss per every 10 mini-batch
        for idx, data in enumerate(train_loader):
            s, p, v = data
            if cuda:
                s, p, v = s.cuda().float(), p.cuda().float(), v.cuda().float()
            optimizer.zero_grad()
            # p_pred = Torch.Size([batch_size,73*8*8]), v_pred = Torch.Size([batch_size,1])
            p_pred, v_pred = nnet(s.float())
            # I don't think this next step is necessary but doesn't hurt to check
            if cuda:
                p_pred, v_pred = p_pred.cuda().float(), v_pred.cuda().float()
            loss = criterion(p_pred, p.float(), v_pred, v.float())
            loss.backward()
            total_loss += loss.item()
            if idx % 10 == 9:
                print(
                    f"Epoch {epoch + 1}/{epoch_end} complete. Total loss is {total_loss/10}.")
                losses_per_batch.append(total_loss)
                total_loss = 0
        avg_loss_per_epoch.append(
            sum(losses_per_batch)/(len(losses_per_batch)+1))

    # add graph
    fig = plt.figure()
    ax = fig.add_subplot(222)
    ax.scatter([e for e in range(1, epoch_end+1, 1)], avg_loss_per_epoch)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss per batch")
    ax.set_title("Loss vs Epoch")
    print('Finished Training')

    plt.savefig(os.path.join("./model_data/", "Loss_vs_Epoch_%s.png" %
                datetime.datetime.today().strftime("%Y-%m-%d")))
