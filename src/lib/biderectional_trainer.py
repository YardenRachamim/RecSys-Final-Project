import os
import lib
import time
import torch
import numpy as np
from tqdm import tqdm
import copy


class BiTrainer(object):
    def __init__(self, model, train_data, eval_data, optim, use_cuda, loss_func, batch_size, args):
        self.model_f = model
        self.model_b = copy.deepcopy(model)

        self.train_data = train_data
        self.eval_data = eval_data
        self.optim = optim
        self.loss_func = loss_func
        self.evaluation = lib.Evaluation(self.model_f, self.loss_func, use_cuda, k = args.k_eval)
        self.device = torch.device('cuda' if use_cuda else 'cpu')
        self.batch_size = batch_size
        self.args = args

    def train(self, start_epoch, end_epoch, start_time=None):
        if start_time is None:
            self.start_time = time.time()
        else:
            self.start_time = start_time

        for epoch in range(start_epoch, end_epoch + 1):
            st = time.time()
            print('Start Epoch #', epoch)
            train_loss = self.train_epoch(epoch)
            loss, recall, mrr = self.evaluation.eval(self.eval_data, self.batch_size)


            print("Epoch: {}, train loss: {:.4f}, loss: {:.4f}, recall: {:.4f}, mrr: {:.4f}, time: {}".format(epoch, train_loss, loss, recall, mrr, time.time() - st))
            checkpoint = {
                'model': self.model_f,
                'model_b': self.model_b,
                'args': self.args,
                'epoch': epoch,
                'optim': self.optim,
                'loss': loss,
                'recall': recall,
                'mrr': mrr
            }
            model_name = os.path.join(self.args.checkpoint_dir, "model_{0:05d}.pt".format(epoch))
            torch.save(checkpoint, model_name)
            print("Save model as %s" % model_name)


    def train_epoch(self, epoch):
        self.model_f.train()
        self.model_b.train()
        losses = []

        def reset_hidden(hidden, mask):
            """Helper function that resets hidden state when some sessions terminate"""
            if len(mask) != 0:
                hidden[:, mask, :] = 0
            return hidden

        hidden_f = self.model_f.init_hidden()
        hidden_b = self.model_b.init_hidden()
        dataloader = lib.DataLoader(self.train_data, self.batch_size)
        #for ii,(data,label) in tqdm(enumerate(train_dataloader),total=len(train_data)):
        for ii, (input, target, mask) in tqdm(enumerate(dataloader), total=len(dataloader.dataset.df) // dataloader.batch_size, miniters = 1000):
            input = input.to(self.device)
            target = target.to(self.device)
            self.optim.zero_grad()
            hidden_f = reset_hidden(hidden_f, mask).detach()
            hidden_b = reset_hidden(hidden_b, mask).detach()
            logit_f, hidden_f = self.model_f(input, hidden_f)
            logit_b, hidden_b = self.model_b(target, hidden_b)

            # output sampling
            logit_sampled_f = logit_f[:, target.view(-1)]
            logit_sampled_b = logit_b[:, input.view(-1)]
            loss_f = self.loss_func(logit_sampled_f)
            loss_b = self.loss_func(logit_sampled_b)
            loss = loss_f + loss_b
            losses.append(loss.item())
            loss.backward()
            self.optim.step()

        mean_losses = np.mean(losses)
        return mean_losses