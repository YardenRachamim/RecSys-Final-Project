import numpy as np
import torch
from tqdm import tqdm

from data.dataloaders import GRU4Rec15Loader


class Evaluation(object):
    def __init__(self, model, loss_func, use_cuda, k=20):
        self.model = model
        self.loss_func = loss_func
        self.topk = k
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')

    def eval(self, eval_data, batch_size):
        self.model.eval()
        losses = []
        recalls = []
        mrrs = []
        # TODO: should be more generic
        dataloader = GRU4Rec15Loader(eval_data, batch_size)

        with torch.no_grad():
            self.model = self.model.to(self.device)
            hidden = self.model.init_hidden()
            hidden = hidden.to(self.device)
            for ii, (input, target, mask) in tqdm(enumerate(dataloader),
                                                  total=len(dataloader),
                                                  miniters=1000):
                # for input, target, mask in dataloader:
                input = input.to(self.device)
                target = target.to(self.device)
                logit, hidden = self.model(input, hidden)
                logit_sampled = logit[:, target.view(-1)]
                loss = self.loss_func(logit_sampled)
                recall, mrr = evaluate(logit, target, k=self.topk)

                # torch.Tensor.item() to get a Python number from a tensor containing a single value
                losses.append(loss.item())
                recalls.append(recall)
                mrrs.append(mrr)
        mean_losses = np.mean(losses)
        mean_recall = np.mean(recalls)
        mean_mrr = np.mean(mrrs)

        return mean_losses, mean_recall, mean_mrr


def get_recall(indices, targets):  # recall --> wether next item in session is within top K=20 recommended items or not
    """
    Calculates the recall score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        recall (float): the recall score
    """
    targets = targets.view(-1, 1).expand_as(indices)
    hits = (targets == indices).nonzero()
    if len(hits) == 0:
        return 0
    n_hits = (targets == indices).nonzero()[:, :-1].size(0)
    recall = float(n_hits) / targets.size(0)
    return recall


def get_mrr(indices, targets):  # Mean Receiprocal Rank --> Average of rank of next item in the session.
    """
    Calculates the MRR score for the given predictions and targets
    Args:
        indices (Bxk): torch.LongTensor. top-k indices predicted by the model.
        targets (B): torch.LongTensor. actual target indices.
    Returns:
        mrr (float): the mrr score
    """
    tmp = targets.view(-1, 1)
    targets = tmp.expand_as(indices)
    hits = (targets == indices).nonzero()
    ranks = hits[:, -1] + 1
    ranks = ranks.float()
    rranks = torch.reciprocal(ranks)
    mrr = torch.sum(rranks).data / targets.size(0)
    return mrr


def evaluate(indices, targets, k=20):
    """
    Evaluates the model using Recall@K, MRR@K scores.

    Args:
        logits (B,C): torch.LongTensor. The predicted logit for the next items.
        targets (B): torch.LongTensor. actual target indices.

    Returns:
        recall (float): the recall score
        mrr (float): the mrr score
    """
    _, indices = torch.topk(indices, k, -1)
    recall = get_recall(indices, targets)
    mrr = get_mrr(indices, targets)

    return recall, mrr
