import os
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm

# from torch_training.eval import Evaluation
from data.datasets import Rec15DataSet
from data.dataloaders import GRU4Rec15Loader
from torch_training.optim import Optimizer
from torch_modules.loss_functions import LossFunction
from configuration import config


class Trainer(object):
    # def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, batch_size, args):
    def __init__(self, model: nn.Module,
                 train_dataset: Rec15DataSet,
                 optim: Optimizer,
                 use_cuda: bool,
                 loss_func: LossFunction,
                 batch_size: int):
        self.model = model
        self.train_data = train_dataset
        # self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        # self.evaluation = Evaluation(self.model, self.loss_func, use_cuda, k=args['k_eval'])
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.batch_size = batch_size
        # self.args = args

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            print('Start Epoch #', epoch)
            train_loss = self.train_epoch(epoch)
            # loss, recall, mrr = self.evaluation.eval(self.eval_data, self.batch_size)

            # print("Epoch: {}, train loss: {:.4f}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, train_loss, loss, recall, mrr, time.time() - st))
            checkpoint = {
                'model': self.model,
                # 'args': self.args,
                'epoch': epoch,
                'optim': self.optim,
                # 'loss': loss,
                # 'recall': recall,
                # 'mrr': mrr
            }
            model_name = config['models_path'] / "model_{0:05d}.pt".format(epoch)
            # model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)

    def train_epoch(self, epoch):
        self.model.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden

        hidden = self.model.init_hidden()
        # TODO: need to be more generic
        dataloader = GRU4Rec15Loader(self.train_data, self.batch_size)

        # for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data)):
        for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader), miniters=1000):
            input = input.to(self.device)
            target = target.to(self.device)
            self.optim.zero_grad()
            hidden = reset_hidden(hidden, mask).detach()
            logit, hidden = self.model(input, hidden)

            # output sampling
            logit_sampled = logit[:, target.view(-1)]
            loss = self.loss_func(logit_sampled)
            losses.append(loss.item())

            loss.backward()
            self.optim.step()

        mean_losses = np.mean(losses)

        return mean_losses
