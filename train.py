import random
import argparse
import os
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
import torch.nn as nn
import torchvision.utils as utils
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
from model_compat import *
from functions import SIMSE, MSE, L1_Loss
from test import test
from torch import autograd
import pdb
import torchvision.utils as vutils
#import h5py
from torch.nn.utils import clip_grad_norm
######################
# params             #
######################
parser = argparse.ArgumentParser(description='')
parser.add_argument('--lr', dest='lr', type=float, default=0.008, help='initial learning rate for adam')
parser.add_argument('--decoder_lr', dest='decoder_lr', type=float, default=0.001, help='initial learning rate for decoder adam')
parser.add_argument('--vae_weight', dest='vae_weight', type=float, default=2, help='vae_weight')
parser.add_argument('--class_transfer_weight', dest='class_transfer_weight', type=float, default=0.1, help='class_transfer_weight')
parser.add_argument('--encoder_class_transfer_weight', dest='encoder_class_transfer_weight', type=float, default=0.6, help='encoder_class_transfer_weight')
parser.add_argument('--latent_transfer_weight', dest='latent_transfer_weight', type=float, default=0.1, help='latent_transfer_weight')
parser.add_argument('--l2_weight', dest='l2_weight', type=float, default=10, help='vae_weight')
parser.add_argument('--trans_gan_weight', dest='trans_gan_weight', type=float, default=0.001, help='trans_gan_weight')
parser.add_argument('--gan_weight', dest='gan_weight', type=float, default=0.001, help='gan_weight')
parser.add_argument('--transfer_epoch', dest='transfer_epoch', type=int, default=100, help='transfer_epoch')
parser.add_argument('--file_name', dest='file_name', default='photo', help='photo art_painting cartoon sketch')
parser.add_argument('--mean_weight', dest='mean_weight', type=float, default=0, help='mean_weight')
args = parser.parse_args()

#model_root = '.'
model_root = './model'
cuda = True
cudnn.benchmark = True
lr = args.lr
batch_size = 100
image_size = 28
n_epoch = 150
step_decay_weight = 0.1
lr_decay_step = 8000
weight_decay = 0.0001
l2_weight = args.l2_weight
momentum = 0.9
rec_mode = 'all'
file_name = args.file_name
print('lr:',lr,'\n')
print('rec_mode:',rec_mode,'\n')

manual_seed = random.randint(1, 10000)
random.seed(manual_seed)
torch.manual_seed(manual_seed)

#######################
# load data           #
#######################
if file_name == 'photo':
    train_0 = np.load('./dataset/PACS_part/train_feature_art_painting.npz')
    domain_data_0 = train_0['x_4096']
    domain_class_label_0 = train_0['class_y']
    train_1 = np.load('./dataset/PACS_part/train_feature_cartoon.npz')
    domain_data_1 = train_1['x_4096']
    domain_class_label_1 = train_1['class_y']
    train_2 = np.load('./dataset/PACS_part/train_feature_sketch.npz')
    domain_data_2 = train_2['x_4096']
    domain_class_label_2 = train_2['class_y']
 
    val_0 = np.load('./dataset/PACS_part/val_feature_art_painting.npz')
    val_data_0 = val_0['x_4096']
    val_label_0 = val_0['class_y']
    val_1 = np.load('./dataset/PACS_part/val_feature_cartoon.npz')
    val_data_1 = val_1['x_4096']
    val_label_1 = val_1['class_y']
    val_2 = np.load('./dataset/PACS_part/val_feature_sketch.npz')
    val_data_2 = val_2['x_4096']
    val_label_2 = val_2['class_y']

    domain_label_0 = np.zeros(domain_class_label_0.shape[0])
    domain_label_1 = np.zeros(domain_class_label_1.shape[0])+1
    domain_label_2 = np.zeros(domain_class_label_2.shape[0])+2
    test_0 = np.load('./dataset/PACS_part/test_feature_photo.npz')
    test_data = test_0['x_4096']
    test_class_label = test_0['class_y']
else:
    print('Not find source dataset')

val_data = np.concatenate((val_data_0,val_data_1,val_data_2),axis=0)
val_label = np.concatenate((val_label_0,val_label_1,val_label_2),axis=0)
del val_data_0,val_data_1,val_data_2,val_label_0,val_label_1,val_label_2

domain_shape = max(domain_data_0.shape[0],domain_data_1.shape[0],domain_data_2.shape[0])
num_batch = int(domain_shape / batch_size + 1)
num = num_batch * batch_size
#pdb.set_trace()
domain0_shape = domain_label_0.shape[0]
domain1_shape = domain_label_1.shape[0]
domain2_shape = domain_label_2.shape[0]


while (num - domain0_shape) > 0:
    to_pad = num - domain_data_0.shape[0]
    domain_data_0 = np.concatenate((domain_data_0,domain_data_0[:to_pad,:]),axis=0)
    domain_class_label_0 = np.concatenate((domain_class_label_0,domain_class_label_0[:to_pad]),axis=0)
    domain_label_0 = np.concatenate((domain_label_0,domain_label_0[:to_pad]),axis=0)
    domain0_shape = domain_label_0.shape[0]

while (num - domain1_shape) > 0:
    to_pad = num - domain_data_1.shape[0]
    domain_data_1 = np.concatenate((domain_data_1,domain_data_1[:to_pad,:]),axis=0)
    domain_class_label_1 = np.concatenate((domain_class_label_1,domain_class_label_1[:to_pad]),axis=0)
    domain_label_1 = np.concatenate((domain_label_1,domain_label_1[:to_pad]),axis=0)
    domain1_shape = domain_label_1.shape[0]

while (num - domain2_shape) > 0:
    to_pad = num - domain_data_2.shape[0]
    domain_data_2 = np.concatenate((domain_data_2,domain_data_2[:to_pad,:]),axis=0)
    domain_class_label_2 = np.concatenate((domain_class_label_2,domain_class_label_2[:to_pad]),axis=0)
    domain_label_2 = np.concatenate((domain_label_2,domain_label_2[:to_pad]),axis=0)
    domain2_shape = domain_label_2.shape[0]

    
#####################
#  load model       #
#####################

my_net = DSN()
#my_source_net = DSN()

#####################
# setup optimizer   #
#####################
def name_in_lst(name,lst):
    if lst is None:
        return False
    for ele in lst:
        if ele in name:
            return True
        return False

def get_param(model,encoder, decoder_0, decoder_1, decoder_2, discri_0, discri_1, discri_2):
    encoder_list = []
    decoder_0_list = []
    decoder_1_list = []
    decoder_2_list = []
    discri_0_list = []
    discri_1_list = []
    discri_2_list = []
    for name,param  in  model.named_parameters():
        if name_in_lst(name, ['decoder_0']):
            decoder_0.append(param)
            decoder_0_list.append(name)
        elif name_in_lst(name, ['discri_0']):
            discri_0.append(param)
            discri_0_list.append(name)
        elif name_in_lst(name, ['decoder_1']):
            decoder_1.append(param)
            decoder_1_list.append(name)
        elif name_in_lst(name, ['discri_1']):
            discri_1.append(param)
            discri_1_list.append(name)
        elif name_in_lst(name, ['decoder_2']):
            decoder_2.append(param)
            decoder_2_list.append(name)
        elif name_in_lst(name, ['discri_2']):
            discri_2.append(param)
            discri_2_list.append(name)
        else:
            encoder.append(param)
            encoder_list.append(name)
    print(encoder_list)
    print(decoder_0_list)
    print(decoder_1_list)
    print(decoder_2_list)
    print(discri_0_list)
    print(discri_1_list)
    print(discri_2_list)
    return encoder, decoder_0, decoder_1, decoder_2, discri_0, discri_1, discri_2


def exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = init_lr * (step_decay_weight ** (step / lr_decay_step))

#    if step % lr_decay_step == 0:
#        print('learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer

def d_exp_lr_scheduler(optimizer, step, init_lr=lr, lr_decay_step=lr_decay_step, step_decay_weight=step_decay_weight):

    # Decay learning rate by a factor of step_decay_weight every lr_decay_step
    current_lr = 0.001 * (0.1 ** (step / 20))

#    if step % lr_decay_step == 0:
#        print('learning rate is set to %f' % current_lr)

    for param_group in optimizer.param_groups:
        param_group['lr'] = current_lr

    return optimizer


encoder = []
decoder_0 = []
discri_0 = []
decoder_1 = []
discri_1 = []
decoder_2 = []
discri_2 = []
encoder, decoder_0, decoder_1, decoder_2, discri_0, discri_1, discri_2 = get_param(my_net,encoder,decoder_0,discri_0,decoder_1,discri_1,decoder_2,discri_2)

#criterion = nn.MSELoss(size_average=False)
#criterion.cuda()
#target_optimizer = optim.SGD(my_target_net.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer_encoder = optim.SGD(encoder, lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer_decoder_0 = optim.SGD(decoder_0, lr=args.decoder_lr, momentum=momentum, weight_decay=weight_decay)
optimizer_decoder_1 = optim.SGD(decoder_1, lr=args.decoder_lr, momentum=momentum, weight_decay=weight_decay)
optimizer_decoder_2 = optim.SGD(decoder_2, lr=args.decoder_lr, momentum=momentum, weight_decay=weight_decay)
optimizer_discri_0 = optim.SGD(discri_0, lr=0.001, momentum=momentum, weight_decay=weight_decay)
optimizer_discri_1 = optim.SGD(discri_1, lr=0.001, momentum=momentum, weight_decay=weight_decay)
optimizer_discri_2 = optim.SGD(discri_2, lr=0.001, momentum=momentum, weight_decay=weight_decay)

loss_classification = torch.nn.CrossEntropyLoss()
loss_l2 = MSE()
loss_bce = torch.nn.BCELoss()
loss_l1 = L1_Loss()
if cuda:
    #my_target_net = my_target_net.cuda()
    my_net = my_net.cuda()
    loss_classification = loss_classification.cuda()
    loss_l2 = loss_l2.cuda()
    loss_bce = loss_bce.cuda()
    loss_l1 = loss_l1.cuda()
for p in my_net.parameters():
    p.requires_grad = True
    #q.requires_grad = True
#############################
# training network          #
#############################

class_mean = np.zeros([7,1024])
indexs = np.arange(domain_data_0.shape[0])
current_step = 0
real_label = 1
fake_label = 0
label = Variable(torch.FloatTensor(batch_size).long().cuda())
class_mean = torch.from_numpy(class_mean).float()
class_mean = class_mean.cuda()
class_mean = Variable(class_mean, requires_grad=True)
mean_optimizer = optim.SGD([class_mean], lr=lr, momentum=momentum, weight_decay=weight_decay)

for epoch in range(n_epoch):

    np.random.shuffle(indexs)
    i = 0

    while i < num_batch:

        ###################################
        # target data training            #
        ###################################
        #pdb.set_trace()
        batch_data_0 = domain_data_0[indexs[i * batch_size:(i+1)*batch_size],:]
        batch_data_1 = domain_data_1[indexs[i * batch_size:(i+1)*batch_size],:]
        batch_data_2 = domain_data_2[indexs[i * batch_size:(i+1)*batch_size],:]

        batch_class_label_0 = domain_class_label_0[indexs[i * batch_size:(i+1)*batch_size]]
        batch_class_label_1 = domain_class_label_1[indexs[i * batch_size:(i+1)*batch_size]]
        batch_class_label_2 = domain_class_label_2[indexs[i * batch_size:(i+1)*batch_size]]
        #pdb.set_trace()
        batch_data_0 = torch.from_numpy(batch_data_0)
        batch_data_1 = torch.from_numpy(batch_data_1)
        batch_data_2 = torch.from_numpy(batch_data_2)

        batch_class_label_0 = torch.from_numpy(batch_class_label_0)
        batch_class_label_1 = torch.from_numpy(batch_class_label_1)
        batch_class_label_2 = torch.from_numpy(batch_class_label_2)
        
        my_net.zero_grad()
        #class_mean.zero_grad()
        loss = 0

        if cuda:
            batch_data_0 = batch_data_0.cuda()
            batch_data_1 = batch_data_1.cuda()
            batch_data_2 = batch_data_2.cuda()

            batch_class_label_0 = batch_class_label_0.long().cuda()
            batch_class_label_1 = batch_class_label_1.long().cuda()
            batch_class_label_2 = batch_class_label_2.long().cuda()

        data_0_train = Variable(batch_data_0)
        data_1_train = Variable(batch_data_1)
        data_2_train = Variable(batch_data_2)

        batch_class_label_0_train = Variable(batch_class_label_0)
        batch_class_label_1_train = Variable(batch_class_label_1)
        batch_class_label_2_train = Variable(batch_class_label_2)

        #new_image_0_label = batch_class_label_0_train
        #new_image_1_label = batch_class_label_1_train
        #new_image_2_label = batch_class_label_2_train

        my_net.zero_grad()
        loss = 0
        #pdb.set_trace()
        result,transfer_pred = my_net(input_data_0 = data_0_train, input_data_1 = data_1_train,input_data_2 = data_2_train,input_label_0 = batch_class_label_0_train,input_label_1 = batch_class_label_1_train,input_label_2 = batch_class_label_2_train, class_number = 7, phase = 'Train',class_mean = class_mean)
        pred_class_label_0 = result[0]
        pred_class_label_1 = result[1]
        pred_class_label_2 = result[2]
        latent_loss = result[7]
        recon_0_img = result[9]
        recon_1_img = result[10]
        recon_2_img = result[11]
        discri_fake_0 = result[12]
        discri_fake_1 = result[13]
        discri_fake_2 = result[14]
        discri_real_0 = result[15]
        discri_real_1 = result[16]
        discri_real_2 = result[17]
        recon_1_0_img = result[18]
        recon_2_0_img = result[19]
        recon_0_1_img = result[20]
        recon_2_1_img = result[21]
        recon_0_2_img = result[22]
        recon_1_2_img = result[23]
        mean_loss = result[24]
        discri_fake_0_1 = result[25]
        discri_fake_0_2 = result[26]
        discri_fake_1_0 = result[27]
        discri_fake_1_2 = result[28]
        discri_fake_2_0 = result[29]
        discri_fake_2_1 = result[30]
        loss -= args.mean_weight * mean_loss
        latent_transfer_loss = args.latent_transfer_weight * transfer_pred[0]
        pred_class_1_0 = transfer_pred[1]
        pred_class_2_0 = transfer_pred[2]
        pred_class_0_1 = transfer_pred[3]
        pred_class_2_1 = transfer_pred[4]
        pred_class_0_2 = transfer_pred[5]
        pred_class_1_2 = transfer_pred[6]

        loss += args.vae_weight * latent_loss
        class_classification_0 = loss_classification(pred_class_label_0, batch_class_label_0_train)
        loss += class_classification_0

        class_classification_1 = loss_classification(pred_class_label_1, batch_class_label_1_train)
        loss += class_classification_1

        class_classification_2 = loss_classification(pred_class_label_2, batch_class_label_2_train)
        loss += class_classification_2

        class_classification = class_classification_0 + class_classification_1 + class_classification_2

        loss.backward(retain_graph = True)
        optimizer = exp_lr_scheduler(optimizer = optimizer_encoder, step=current_step)
        optimizer.step()

        mean_optimizer = exp_lr_scheduler(optimizer = mean_optimizer, step=current_step)
        mean_optimizer.step()
        if i % 2 == 0 or epoch < args.transfer_epoch:
            #pdb.set_trace()  
            label.fill_(fake_label)
            errD_0_fake = args.gan_weight * loss_classification(discri_fake_0, label)
            errD_0_fake.backward(retain_graph = True)

            errD_1_0_fake = args.trans_gan_weight * loss_classification(discri_fake_1_0, label)
            errD_1_0_fake.backward(retain_graph = True)

            errD_2_0_fake = args.trans_gan_weight * loss_classification(discri_fake_2_0, label)
            errD_2_0_fake.backward(retain_graph = True)

            label.fill_(real_label)
            errD_0_real = args.gan_weight * loss_classification(discri_real_0, label)
            errD_0_real.backward(retain_graph = True)
            errD_0_loss = errD_0_fake + errD_0_real 
    
            label.fill_(fake_label)
            errD_1_fake = args.gan_weight * loss_classification(discri_fake_1, label)
            errD_1_fake.backward(retain_graph = True)

            errD_0_1_fake = args.trans_gan_weight * loss_classification(discri_fake_0_1, label)
            errD_0_1_fake.backward(retain_graph = True)

            errD_2_1_fake = args.trans_gan_weight * loss_classification(discri_fake_2_1, label)
            errD_2_1_fake.backward(retain_graph = True)

            label.fill_(real_label)
            errD_1_real = args.gan_weight * loss_classification(discri_real_1, label)
            errD_1_real.backward(retain_graph = True)
            errD_1_loss = errD_1_fake + errD_1_real
    
            label.fill_(fake_label)
            errD_2_fake = args.gan_weight * loss_classification(discri_fake_2, label)
            errD_2_fake.backward(retain_graph = True)

            errD_0_2_fake = args.trans_gan_weight * loss_classification(discri_fake_0_2, label)
            errD_0_2_fake.backward(retain_graph = True)

            errD_1_2_fake = args.trans_gan_weight * loss_classification(discri_fake_1_2, label)
            errD_1_2_fake.backward(retain_graph = True)

            label.fill_(real_label)
            errD_2_real = args.gan_weight * loss_classification(discri_real_2, label)
            errD_2_real.backward(retain_graph = True)
            errD_2_loss = errD_2_fake + errD_2_real

            optimizer_discri_0.step()
            optimizer_discri_1.step()
            optimizer_discri_2.step()

            mean_optimizer.zero_grad()
            my_net.zero_grad()
    

            label.fill_(real_label)
            errG_0_loss = args.gan_weight * loss_classification(discri_fake_0, label)
            errG_1_loss = args.gan_weight * loss_classification(discri_fake_1, label)
            errG_2_loss = args.gan_weight * loss_classification(discri_fake_2, label)
            errG_0_1_loss = args.trans_gan_weight * loss_classification(discri_fake_0_1, label)
            errG_0_2_loss = args.trans_gan_weight * loss_classification(discri_fake_0_2, label)
            errG_1_0_loss = args.trans_gan_weight * loss_classification(discri_fake_1_0, label)
            errG_1_2_loss = args.trans_gan_weight * loss_classification(discri_fake_1_2, label)
            errG_2_0_loss = args.trans_gan_weight * loss_classification(discri_fake_2_0, label)
            errG_2_1_loss = args.trans_gan_weight * loss_classification(discri_fake_2_1, label)
 
            recon_loss_0 = l2_weight * loss_l2(data_0_train, recon_0_img)
            recon_loss_1 = l2_weight * loss_l2(data_1_train, recon_1_img)
            recon_loss_2 = l2_weight * loss_l2(data_2_train, recon_2_img)
    
            class_classification_0_1 = args.class_transfer_weight * loss_classification(pred_class_0_1, batch_class_label_0_train)
            class_classification_0_2 = args.class_transfer_weight * loss_classification(pred_class_0_2, batch_class_label_0_train)
    
    
            class_classification_1_0 = args.class_transfer_weight * loss_classification(pred_class_1_0, batch_class_label_1_train)
            class_classification_1_2 = args.class_transfer_weight * loss_classification(pred_class_1_2, batch_class_label_1_train)
    
    
            class_classification_2_0 = args.class_transfer_weight * loss_classification(pred_class_2_0, batch_class_label_2_train)
            class_classification_2_1 = args.class_transfer_weight * loss_classification(pred_class_2_1, batch_class_label_2_train)
    
            latent_transfer_loss.backward(retain_graph = True)
            errG_0_loss.backward(retain_graph = True)
            errG_1_loss.backward(retain_graph = True)
            errG_2_loss.backward(retain_graph = True)
            errG_0_1_loss.backward(retain_graph = True)
            errG_0_2_loss.backward(retain_graph = True)
            errG_1_0_loss.backward(retain_graph = True)
            errG_1_2_loss.backward(retain_graph = True)
            errG_2_0_loss.backward(retain_graph = True)
            errG_2_1_loss.backward(retain_graph = True)
            recon_loss_0.backward(retain_graph = True)
            recon_loss_1.backward(retain_graph = True)
            recon_loss_2.backward(retain_graph = True)
            class_classification_0_1.backward(retain_graph = True)
            class_classification_0_2.backward(retain_graph = True)
            class_classification_1_0.backward(retain_graph = True)
            class_classification_1_2.backward(retain_graph = True)
            class_classification_2_0.backward(retain_graph = True)
            class_classification_2_1.backward(retain_graph = True)
   
            optimizer_decoder_0.step()
            optimizer_decoder_1.step()
            optimizer_decoder_2.step()
            mean_optimizer.zero_grad()
        else:
            class_classification_0_1 = args.encoder_class_transfer_weight * loss_classification(pred_class_0_1, batch_class_label_0_train)
            class_classification_0_2 = args.encoder_class_transfer_weight * loss_classification(pred_class_0_2, batch_class_label_0_train)


            class_classification_1_0 = args.encoder_class_transfer_weight * loss_classification(pred_class_1_0, batch_class_label_1_train)
            class_classification_1_2 = args.encoder_class_transfer_weight * loss_classification(pred_class_1_2, batch_class_label_1_train)


            class_classification_2_0 = args.encoder_class_transfer_weight * loss_classification(pred_class_2_0, batch_class_label_2_train)
            class_classification_2_1 = args.encoder_class_transfer_weight * loss_classification(pred_class_2_1, batch_class_label_2_train)

            latent_transfer_loss.backward(retain_graph = True)
            class_classification_0_1.backward(retain_graph = True)
            class_classification_0_2.backward(retain_graph = True)
            class_classification_1_0.backward(retain_graph = True)
            class_classification_1_2.backward(retain_graph = True)
            class_classification_2_0.backward(retain_graph = True)
            class_classification_2_1.backward(retain_graph = True)
            optimizer = exp_lr_scheduler(optimizer = optimizer_encoder, step=current_step)
            optimizer.step()
    
            mean_optimizer = exp_lr_scheduler(optimizer = mean_optimizer, step=current_step)
            mean_optimizer.step()
            mean_optimizer.zero_grad()
        i += 1
        current_step += 1
    print('source_classification: %f,latent_loss: %f, errG_0_loss: %f ,errG_1_loss: %f ,errG_2_loss: %f , errG_0_1_loss: %f ,errG_0_2_loss: %f ,errG_1_0_loss: %f , errG_1_2_loss: %f ,errG_2_0_loss: %f ,errG_2_1_loss: %f , errD_0_loss: %f ,errD_1_loss: %f ,errD_2_loss: %f ,errD_0_1_loss: %f ,errD_0_2_loss: %f ,errD_1_0_loss: %f ,errD_1_2_loss: %f ,errD_2_0_loss: %f ,errD_2_1_loss: %f , recon_0_loss: %f ,recon_1_loss: %f ,recon_2_loss: %f'%(class_classification.data.cpu().numpy(),latent_loss.data.cpu().numpy(),errG_0_loss.data.cpu().numpy(),errG_1_loss.data.cpu().numpy(),errG_2_loss.data.cpu().numpy(),errG_0_1_loss.data.cpu().numpy(),errG_0_2_loss.data.cpu().numpy(),errG_1_0_loss.data.cpu().numpy(),errG_1_2_loss.data.cpu().numpy(),errG_2_0_loss.data.cpu().numpy(),errG_2_1_loss.data.cpu().numpy(),errD_0_loss.data.cpu().numpy(),errD_1_loss.data.cpu().numpy(),errD_2_loss.data.cpu().numpy(),errD_0_1_fake.data.cpu().numpy(),errD_0_2_fake.data.cpu().numpy(),errD_1_0_fake.data.cpu().numpy(),errD_1_2_fake.data.cpu().numpy(),errD_2_0_fake.data.cpu().numpy(),errD_2_1_fake.data.cpu().numpy(),recon_loss_0.data.cpu().numpy(),recon_loss_1.data.cpu().numpy(),recon_loss_2.data.cpu().numpy()))
#    if epoch % 900 == 0:
#        torch.save(my_net.state_dict(), model_root + '/dsn_mnist_epoch_' + str(epoch) +'_'+ str(args.vae_weight) + '.pth')
    print('all')
    my_net.eval()
    test(epoch=epoch, name='val_'+file_name,test_data = val_data, test_class_label = val_label, rec_mode = 'val', model_root = model_root, my_net=my_net)
    test(epoch=epoch, name='test_'+file_name,test_data = test_data, test_class_label = test_class_label, rec_mode = 'all', model_root = model_root, my_net=my_net)
    my_net.train(mode=True)
print('done')
