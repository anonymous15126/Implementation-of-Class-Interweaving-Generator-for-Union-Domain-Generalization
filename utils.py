import torch
from torch.autograd import Variable
from torch.autograd import Function
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from functools import partial
import numpy as np
import torch.nn as nn
import pdb
from itertools import combinations
import torch.nn.functional as F
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, p):
        ctx.p = p

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.p

        return output, None

def pairwise_distance(x, y):

    if not len(x.shape) == len(y.shape) == 2:
        raise ValueError('Both inputs should be matrices.')

    if x.shape[1] != y.shape[1]:
        raise ValueError('The number of features should be the same.')

    x = x.view(x.shape[0], x.shape[1], 1)
    y = torch.transpose(y, 0, 1)
    output = torch.sum((x - y) ** 2, 1)
    output = torch.transpose(output, 0, 1)

    return output

def gaussian_kernel_matrix(x, y, sigmas):

    sigmas = sigmas.view(sigmas.shape[0], 1)
    beta = 1. / (2. * sigmas)
    dist = pairwise_distance(x, y).contiguous()
    dist_ = dist.view(1, -1)
    s = torch.matmul(beta, dist_)

    return torch.sum(torch.exp(-s), 0).view_as(dist)

def maximum_mean_discrepancy(x, y, kernel= gaussian_kernel_matrix):

    cost = torch.mean(kernel(x, x))
    cost += torch.mean(kernel(y, y))
    cost -= 2 * torch.mean(kernel(x, y))

    return cost

def mmd_loss(source_features, target_features):

    sigmas = [
        1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 5, 10, 15, 20, 25, 30, 35, 100,
        1e3, 1e4, 1e5, 1e6
    ]
    if params.use_gpu:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.cuda.FloatTensor(sigmas))
        )
    else:
        gaussian_kernel = partial(
            gaussian_kernel_matrix, sigmas = Variable(torch.FloatTensor(sigmas))
        )
    loss_value = maximum_mean_discrepancy(source_features, target_features, kernel= gaussian_kernel)
    loss_value = loss_value

    return loss_value

def triplet_loss(features, labels):
    #model.train()
    #emb = model(batch["X"].cuda())
    #y = batch["y"].cuda()
    #pdb.set_trace()    

    #with torch.no_grad():
    triplets = get_triplets(features, labels)
    f_A = features[triplets[:, 0].cuda()]
    f_P = features[triplets[:, 1].cuda()]
    f_N = features[triplets[:, 2].cuda()]

    ap_D = (f_A - f_P).pow(2).sum(1)  # .pow(.5)
    an_D = (f_A - f_N).pow(2).sum(1)  # .pow(.5)
    losses = F.relu(ap_D - an_D + 1.)

    return losses.mean()


def center_loss(tgt_model, batch, src_model, src_centers, tgt_centers, 
                src_kmeans, tgt_kmeans, margin=1):
    # triplets = self.triplet_selector.get_triplets(embeddings, target, embeddings_adv=embeddings_adv)
    # triplets = triplets.cuda()


    #f_N = embeddings_adv[triplets[:, 2]]

    f_N_clf = tgt_model.convnet(batch["X"].cuda()).view(batch["X"].shape[0], -1)
    f_N = tgt_model.fc(f_N_clf.detach())
    
    #est.predict(f_N.cpu().numpy())
    y_src = src_kmeans.predict(f_N.detach().cpu().numpy())
    #ap_distances = (emb_centers[None] - f_N[:,None]).pow(2).min(1)[0].sum(1)
    ap_distances = (src_centers[y_src] - f_N).pow(2).sum(1)
    #ap_distances = (f_C[None] - f_N[:,None]).pow(2).sum(1).sum(1)

    
    #an_distances = 0
    losses = ap_distances.mean()

    # y_tgt = tgt_kmeans.predict(f_N.detach().cpu().numpy())
    # ap_distances = (tgt_centers[y_tgt] - f_N).pow(2).max(1)[0]

    # losses += ap_distances.mean()*0.1

    # f_P = src_model(batch["X"].cuda())
    #an_distances = (f_P - f_N).pow(2).sum(1)
    #losses -= an_distances.mean() * 0.1
  
    return losses



### Triplets Utils

def extract_embeddings(model, dataloader):
    model.eval()
    n_samples = dataloader.batch_size * len(dataloader)
    embeddings = np.zeros((n_samples, model.n_outputs))
    labels = np.zeros(n_samples)
    k = 0

    for images, target in dataloader:
        with torch.no_grad():
            images = images.cuda()            
            embeddings[k:k+len(images)] = model.get_embedding(images).data.cpu().numpy()
            labels[k:k+len(images)] = target.numpy()
            k += len(images)

    return embeddings, labels
    
def get_triplets(embeddings, y):

  margin = 1
  D = pdist(embeddings)
  D = D.cpu()

  y = y.cpu().data.numpy().ravel()
  trip = []

  for label in set(y):
      label_mask = (y == label)
      label_indices = np.where(label_mask)[0]
      if len(label_indices) < 2:
          continue
      neg_ind = np.where(np.logical_not(label_mask))[0]
      
      ap = list(combinations(label_indices, 2))  # All anchor-positive pairs
      ap = np.array(ap)

      ap_D = D[ap[:, 0], ap[:, 1]]
      
      # # GET HARD NEGATIVE
      # if np.random.rand() < 0.5:
      #   trip += get_neg_hard(neg_ind, hardest_negative,
      #                D, ap, ap_D, margin)
      # else:
      trip += get_neg_hard(neg_ind, hardest_negative,
                 D, ap, ap_D, margin)

  if len(trip) == 0:
      ap = ap[0]
      trip.append([ap[0], ap[1], neg_ind[0]])

  trip = np.array(trip)

  return torch.LongTensor(trip)




def pdist(vectors):
    D = -2 * vectors.mm(torch.t(vectors)) 
    D += vectors.pow(2).sum(dim=1).view(1, -1) 
    D += vectors.pow(2).sum(dim=1).view(-1, 1)

    return D


def get_neg_hard(neg_ind, 
                      select_func,
                      D, ap, ap_D, margin):
    trip = []

    for ap_i, ap_di in zip(ap, ap_D):
        loss_values = (ap_di - 
               D[torch.LongTensor(np.array([ap_i[0]])), 
                torch.LongTensor(neg_ind)] + margin)

        loss_values = loss_values.data.cpu().numpy()
        neg_hard = select_func(loss_values)

        if neg_hard is not None:
            neg_hard = neg_ind[neg_hard]
            trip.append([ap_i[0], ap_i[1], neg_hard])

    return trip

def random_neg(loss_values):
    neg_hards = np.where(loss_values > 0)[0]
    return np.random.choice(neg_hards) if len(neg_hards) > 0 else None

def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None

def semihard_negative(loss_values, margin=1):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None

