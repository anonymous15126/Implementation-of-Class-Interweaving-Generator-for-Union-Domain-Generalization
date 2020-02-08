import torch.nn as nn
from functions import ReverseLayerF
import torch
import pdb
from torch.autograd import Variable
from torch import autograd
import numpy as np
def latent_loss(class_mean, mean_0, log_stddev_0, index_0):
    #class_mean_temp_0 = torch.from_numpy(np.tile(class_mean , (len(index_0),1))).float()
    mean_temp_0 = mean_0[index_0,:] - class_mean #Variable(class_mean_temp_0.cuda(), requires_grad=False)
    log_stddev_temp_0 = log_stddev_0[index_0,:]
    stddev_temp_0 = torch.exp(log_stddev_temp_0)
    mean_sq_temp_0 = mean_temp_0 * mean_temp_0
    stddev_sq_temp_0 = stddev_temp_0 * stddev_temp_0
    loss = 0.5 * torch.mean(mean_sq_temp_0 + stddev_sq_temp_0 - torch.log(stddev_sq_temp_0) - 1)
    return loss
class DSN(nn.Module):
    def __init__(self, code_size=4096, n_class=7):
        super(DSN, self).__init__()
        self.code_size = code_size

        ################################
        # shared encoder
        ################################

        self.encoder_fc = nn.Sequential()
        self.encoder_fc.add_module('dropout',nn.Dropout())
        self.encoder_fc.add_module('fc_1', nn.Linear(in_features=code_size, out_features=4096))
        self.encoder_fc.add_module('relu', nn.ReLU(True))

        ######### variation autoencoder ########
        self.encoder_mean = nn.Sequential()
        self.encoder_mean.add_module('fc_se3', nn.Linear(in_features=4096, out_features=1024))
        self.encoder_mean.add_module('ac_se3', nn.ReLU(True))

        self.encoder_stddev = nn.Sequential()
        self.encoder_stddev.add_module('fc_se3', nn.Linear(in_features=4096, out_features=1024))
        self.encoder_stddev.add_module('ac_se3', nn.ReLU(True))

        self.encoder_pred_class = nn.Sequential()

        self.encoder_pred_class.add_module('fc', nn.Linear(in_features=1024, out_features=n_class))

        ######## encoder the hidden layer ########
        self.decoder_0_fc = nn.Sequential()

        self.decoder_0_fc.add_module('fc_se4', nn.Linear(in_features= 1024, out_features=4096))
        self.decoder_0_fc.add_module('bn_se4', nn.BatchNorm1d(4096))
        self.decoder_0_fc.add_module('ac_se4', nn.ReLU(True))

        self.decoder_0_fc.add_module('fc_se3', nn.Linear(in_features=4096, out_features=4096))
        self.decoder_0_fc.add_module('bn_se3', nn.BatchNorm1d(4096))
        self.decoder_0_fc.add_module('ac_se3', nn.Tanh())

        ######## encoder the hidden layer ########
        self.decoder_1_fc = nn.Sequential()
        self.decoder_1_fc.add_module('fc_se4', nn.Linear(in_features= 1024, out_features=4096))
        self.decoder_1_fc.add_module('bn_se4', nn.BatchNorm1d(4096))
        self.decoder_1_fc.add_module('ac_se4', nn.ReLU(True))

        self.decoder_1_fc.add_module('fc_se3', nn.Linear(in_features=4096, out_features=4096))
        self.decoder_1_fc.add_module('bn_se3', nn.BatchNorm1d(4096))
        self.decoder_1_fc.add_module('ac_se3', nn.Tanh())

        ######## encoder the hidden layer ########
        self.decoder_2_fc = nn.Sequential()

        self.decoder_2_fc.add_module('fc_se4', nn.Linear(in_features= 1024, out_features=4096))
        self.decoder_2_fc.add_module('bn_se4', nn.BatchNorm1d(4096))
        self.decoder_2_fc.add_module('ac_se4', nn.ReLU(True))

        self.decoder_2_fc.add_module('fc_se3', nn.Linear(in_features=4096, out_features=4096))
        self.decoder_2_fc.add_module('bn_se3', nn.BatchNorm1d(4096))
        self.decoder_2_fc.add_module('ac_se3', nn.Tanh())


        ######## discriminator ########
        self.discri_0_fc = nn.Sequential()
        self.discri_0_fc.add_module('fc_se3', nn.Linear(in_features=4096, out_features=1024))
        self.discri_0_fc.add_module('ac_se3', nn.ReLU(True))

        ######## encoder the hidden layer ########
        self.discri_0_fc.add_module('dropout', nn.Dropout(0.9))
        self.discri_0_fc.add_module('fc_se4', nn.Linear(in_features= 1024, out_features=1024))
        self.discri_0_fc.add_module('ac_se4', nn.ReLU(True))

        self.discri_0_fc.add_module('fc_se5', nn.Linear(in_features=1024, out_features=2))
        #self.discri_0_fc_3.add_module('sigmoid', nn.Sigmoid())

         ######## discriminator ########
        self.discri_1_fc = nn.Sequential()
        self.discri_1_fc.add_module('fc_se3', nn.Linear(in_features=4096, out_features=1024))
        self.discri_1_fc.add_module('ac_se3', nn.ReLU(True))

        ######## encoder the hidden layer ########
        self.discri_1_fc.add_module('dropout', nn.Dropout(0.9))
        self.discri_1_fc.add_module('fc_se4', nn.Linear(in_features= 1024, out_features=1024))
        self.discri_1_fc.add_module('ac_se4', nn.ReLU(True))

        self.discri_1_fc.add_module('fc_se5', nn.Linear(in_features=1024, out_features=2))
        #self.discri_1_fc_3.add_module('sigmoid', nn.Sigmoid())

        ######## discriminator ########
        self.discri_2_fc = nn.Sequential()
        self.discri_2_fc.add_module('fc_se3', nn.Linear(in_features=4096, out_features=1024))
        self.discri_2_fc.add_module('ac_se3', nn.ReLU(True))

        ######## encoder the hidden layer ########
        self.discri_2_fc.add_module('dropout', nn.Dropout(0.9))
        self.discri_2_fc.add_module('fc_se4', nn.Linear(in_features= 1024, out_features=1024))
        self.discri_2_fc.add_module('ac_se4', nn.ReLU(True))

        self.discri_2_fc.add_module('fc_se5', nn.Linear(in_features=1024, out_features=2))
        #self.discri_2_fc_3.add_module('sigmoid', nn.Sigmoid())
    def latent_loss(self,class_mean, mean_0, log_stddev_0, index_0):
        #class_mean_temp_0 = torch.from_numpy(np.tile(class_mean , (len(index_0),1))).float()
        mean_temp_0 = mean_0[index_0,:] - class_mean#Variable(class_mean_temp_0.cuda(), requires_grad=False)
        log_stddev_temp_0 = log_stddev_0[index_0,:]
        stddev_temp_0 = torch.exp(log_stddev_temp_0)
        mean_sq_temp_0 = mean_temp_0 * mean_temp_0
        stddev_sq_temp_0 = stddev_temp_0 * stddev_temp_0
        loss = 0.5 * torch.mean(mean_sq_temp_0 + stddev_sq_temp_0 - torch.log(stddev_sq_temp_0) - 1)
        return loss


    def forward(self, input_data_0=0,input_data_1=0,input_data_2=0, input_label_0=0, input_label_1=0, input_label_2=0, class_number=10, phase=True, class_mean=0, p=1.0):

        result = []

        code_0 = self.encoder_fc(input_data_0)

        ######## encoder the hidden layer ##########
        mean_0 = self.encoder_mean(code_0)
        log_stddev_0 = self.encoder_stddev(code_0)
        stddev_0 = torch.exp(log_stddev_0)

        code_1 = self.encoder_fc(input_data_1)

        ######## encoder the hidden layer ##########
        mean_1 = self.encoder_mean(code_1)
        log_stddev_1 = self.encoder_stddev(code_1)
        stddev_1 = torch.exp(log_stddev_1)

        code_2 = self.encoder_fc(input_data_2)

        ######## encoder the hidden layer ##########
        mean_2 = self.encoder_mean(code_2)
        log_stddev_2 = self.encoder_stddev(code_2)
        stddev_2 = torch.exp(log_stddev_2)
#        mean_sq = mean * mean
#        stddev_sq = stddev * stddev

        ########### compute the kl loss ###############
#        latent_loss = 0.5 * torch.mean(mean_sq + stddev_sq - torch.log(stddev_sq) - 1)
        latent_loss_temp = 0
        if phase == 'Train':
            label_0_temp = input_label_0.cpu().data.numpy()
            label_1_temp = input_label_1.cpu().data.numpy()
            label_2_temp = input_label_2.cpu().data.numpy()
            count = 0
            for i in range(0, class_number):
                index_0 = np.argwhere(label_0_temp == i)
                index_1 = np.argwhere(label_1_temp == i)
                index_2 = np.argwhere(label_2_temp == i)
                #pdb.set_trace()
                if len(index_0) > 0:
                    latent_loss_0 = self.latent_loss(class_mean[i,:], mean_0, log_stddev_0, index_0)
                    latent_loss_temp = latent_loss_temp + latent_loss_0
                    count = count + 1
                if len(index_1) > 0:
                    latent_loss_1 = self.latent_loss(class_mean[i,:], mean_1, log_stddev_1, index_1)
                    latent_loss_temp = latent_loss_temp + latent_loss_1
                    count = count + 1
                if len(index_2) > 0:
                    latent_loss_2 = self.latent_loss(class_mean[i,:], mean_2, log_stddev_2, index_2)
                    latent_loss_temp = latent_loss_temp + latent_loss_2
                    count = count + 1
            latent_loss = latent_loss_temp / (count)
        else:
            latent_loss = 0

        if phase == 'Train':
            std_z_0 = torch.from_numpy(np.random.normal(0, 1, size=stddev_0.size())).float()
            latent_z_0 = mean_0 + stddev_0 * Variable(std_z_0.cuda(), requires_grad=False)
            pred_class_0 = self.encoder_pred_class(latent_z_0)

            std_z_1 = torch.from_numpy(np.random.normal(0, 1, size=stddev_1.size())).float()
            latent_z_1 = mean_1 + stddev_1 * Variable(std_z_1.cuda(), requires_grad=False)
            pred_class_1 = self.encoder_pred_class(latent_z_1)

            std_z_2 = torch.from_numpy(np.random.normal(0, 1, size=stddev_2.size())).float()
            latent_z_2 = mean_2 + stddev_2 * Variable(std_z_2.cuda(), requires_grad=False)
            pred_class_2 = self.encoder_pred_class(latent_z_2)

        else:
            std_z_1 = torch.from_numpy(np.random.normal(0, 1, size=stddev_0.size())).float()
            latent_z_1 = mean_0 #+ stddev_0 * Variable(std_z_1.cuda(), requires_grad=False)
            pred_class_1 = self.encoder_pred_class(latent_z_1)

            std_z_2 = torch.from_numpy(np.random.normal(0, 1, size=stddev_0.size())).float()
            latent_z_2 = mean_0 #+ stddev_0 * Variable(std_z_2.cuda(), requires_grad=False)
            pred_class_2 = self.encoder_pred_class(latent_z_2)

            std_z_3 = torch.from_numpy(np.random.normal(0, 1, size=stddev_0.size())).float()
            latent_z_3 = mean_0 #+ stddev_0 * Variable(std_z_3.cuda(), requires_grad=False)
            pred_class_3 = self.encoder_pred_class(latent_z_3)

            #std_z_4 = torch.from_numpy(np.random.normal(0, 1, size=stddev.size())).float()
            #latent_z_4 = mean + stddev * Variable(std_z_4.cuda(), requires_grad=False)
            #pred_class_4 = self.encoder_pred_class(latent_z_4)
            latent_z = (latent_z_1 + latent_z_2 + latent_z_3 ) / 3
            pred_class = (pred_class_1 + pred_class_2 + pred_class_3 )/3
 
        #reversed_shared_code = ReverseLayerF.apply(code_0, p)
        #domain_label = self.encoder_pred_domain(reversed_shared_code)

        ######### predict the class #########
        #pred_class = self.encoder_pred_class(latent_z)
        #pred_class = self.encoder_pred_class(encoder_pred_class)
        if phase == 'Train':

            recon_0_img = self.decoder_0_fc(mean_0)
            recon_0_img = (recon_0_img + 1)/2.0  

            recon_1_0_img = self.decoder_0_fc(mean_1)
            recon_1_0_img = (recon_1_0_img + 1)/2.0  


            recon_2_0_img = self.decoder_0_fc(mean_2)
            recon_2_0_img = (recon_2_0_img + 1)/2.0  


            recon_1_img = self.decoder_1_fc(mean_1)
            recon_1_img = (recon_1_img + 1)/2.0  


            recon_0_1_img = self.decoder_1_fc(mean_0)
            recon_0_1_img = (recon_0_1_img + 1)/2.0  


            recon_2_1_img = self.decoder_1_fc(mean_2)
            recon_2_1_img = (recon_2_1_img + 1)/2.0  


            recon_2_img = self.decoder_2_fc(mean_2)
            recon_2_img = (recon_2_img + 1)/2.0  


            recon_0_2_img = self.decoder_2_fc(mean_0)
            recon_0_2_img = (recon_0_2_img + 1)/2.0  


            recon_1_2_img = self.decoder_2_fc(mean_1)
            recon_1_2_img = (recon_1_2_img + 1)/2.0  

            ######## encoder the hidden layer ##########
            code_1_0 = self.encoder_fc(recon_1_0_img)

            mean_1_0 = self.encoder_mean(code_1_0)
            log_stddev_1_0 = self.encoder_stddev(code_1_0)
            stddev_1_0 = torch.exp(log_stddev_1_0)

            ######## encoder the hidden layer ##########    
            code_2_0 = self.encoder_fc(recon_2_0_img)
    
            mean_2_0 = self.encoder_mean(code_2_0)
            log_stddev_2_0 = self.encoder_stddev(code_2_0)
            stddev_2_0 = torch.exp(log_stddev_2_0)
    
            ######## encoder the hidden layer ##########
            code_0_1 = self.encoder_fc(recon_0_1_img)
    
            mean_0_1 = self.encoder_mean(code_0_1)
            log_stddev_0_1 = self.encoder_stddev(code_0_1)
            stddev_0_1 = torch.exp(log_stddev_0_1)
    
            ######## encoder the hidden layer ##########
            code_2_1 = self.encoder_fc(recon_2_1_img)
    
            mean_2_1 = self.encoder_mean(code_2_1)
            log_stddev_2_1 = self.encoder_stddev(code_2_1)
            stddev_2_1 = torch.exp(log_stddev_2_1)
    
            ######## encoder the hidden layer ##########
            code_0_2 = self.encoder_fc(recon_0_2_img)

            mean_0_2 = self.encoder_mean(code_0_2)
            log_stddev_0_2 = self.encoder_stddev(code_0_2)
            stddev_0_2 = torch.exp(log_stddev_0_2)
    
            ######## encoder the hidden layer ##########
            code_1_2 = self.encoder_fc(recon_1_2_img)

            mean_1_2 = self.encoder_mean(code_1_2)
            log_stddev_1_2 = self.encoder_stddev(code_1_2)
            stddev_1_2 = torch.exp(log_stddev_1_2)

        latent_loss_temp = 0 
        transfer_pred = []
        if phase == 'Train':
            label_0_temp = input_label_0.cpu().data.numpy()
            label_1_temp = input_label_1.cpu().data.numpy()
            label_2_temp = input_label_2.cpu().data.numpy()
            count = 0
            for i in range(0, class_number):
                index_0 = np.argwhere(label_0_temp == i)
                index_1 = np.argwhere(label_1_temp == i)
                index_2 = np.argwhere(label_2_temp == i)
                if len(index_0) > 0:
                    latent_loss_0_1 = self.latent_loss(class_mean[i,:], mean_0_1, log_stddev_0_1, index_0)
                    latent_loss_temp = latent_loss_temp + latent_loss_0_1

                    latent_loss_0_2 = self.latent_loss(class_mean[i,:], mean_0_2, log_stddev_0_2, index_0)
                    latent_loss_temp = latent_loss_temp + latent_loss_0_2

                    count = count + 2
                if len(index_1) > 0:
                    latent_loss_1_0 = self.latent_loss(class_mean[i,:], mean_1_0, log_stddev_1_0, index_1)
                    latent_loss_temp = latent_loss_temp + latent_loss_1_0

                    latent_loss_1_2 = self.latent_loss(class_mean[i,:], mean_1_2, log_stddev_1_2, index_1)
                    latent_loss_temp = latent_loss_temp + latent_loss_1_2

                    count = count + 2

                if len(index_2) > 0:
                    latent_loss_2_0 = self.latent_loss(class_mean[i,:], mean_2_0, log_stddev_2_0, index_2)
                    latent_loss_temp = latent_loss_temp + latent_loss_2_0

                    latent_loss_2_1 = self.latent_loss(class_mean[i,:], mean_2_1, log_stddev_2_1, index_2)
                    latent_loss_temp = latent_loss_temp + latent_loss_2_1

                    count = count + 2

            latent_transfer_loss = latent_loss_temp / (count)

            class_mean_1 = class_mean.repeat(1,class_number,1).view(7,-1,1024)
            class_mean_2 = class_mean_1.permute(1,0,2)
            #pdb.set_trace()
            mean_loss = torch.sum(torch.abs(class_mean_1 - class_mean_2) ** 2) / (class_number * class_number * 1024)
            #print(mean_loss.data.cpu().numpy())
            del class_mean_1,class_mean_2

            std_z_1_0 = torch.from_numpy(np.random.normal(0, 1, size=stddev_1_0.size())).float()
            latent_z_1_0 = mean_1_0 + stddev_1_0 * Variable(std_z_1_0.cuda(), requires_grad=False)
            encoder_pred_class_0 = self.encoder_pred_class(latent_z_1_0)

            std_z_2_0 = torch.from_numpy(np.random.normal(0, 1, size=stddev_2_0.size())).float()
            latent_z_2_0 = mean_2_0 + stddev_2_0 * Variable(std_z_2_0.cuda(), requires_grad=False)
            pred_class_2_0 = self.encoder_pred_class(latent_z_2_0)

            std_z_0_1 = torch.from_numpy(np.random.normal(0, 1, size=stddev_0_1.size())).float()
            latent_z_0_1 = mean_0_1 + stddev_0_1 * Variable(std_z_0_1.cuda(), requires_grad=False)
            pred_class_0_1 = self.encoder_pred_class(latent_z_0_1)

            std_z_2_1 = torch.from_numpy(np.random.normal(0, 1, size=stddev_2_1.size())).float()
            latent_z_2_1 = mean_2_1 + stddev_2_1 * Variable(std_z_2_1.cuda(), requires_grad=False)
            pred_class_2_1 = self.encoder_pred_class(latent_z_2_1)

            std_z_0_2 = torch.from_numpy(np.random.normal(0, 1, size=stddev_0_2.size())).float()
            latent_z_0_2 = mean_0_2 + stddev_0_2 * Variable(std_z_0_2.cuda(), requires_grad=False)
            pred_class_0_2 = self.encoder_pred_class(latent_z_0_2)

            std_z_1_2 = torch.from_numpy(np.random.normal(0, 1, size=stddev_1_2.size())).float()
            latent_z_1_2 = mean_1_2 + stddev_1_2 * Variable(std_z_1_2.cuda(), requires_grad=False)
            encoder_pred_class_2 = self.encoder_pred_class(latent_z_1_2)

            transfer_pred.append(latent_transfer_loss)
            transfer_pred.append(encoder_pred_class_0)
            transfer_pred.append(pred_class_2_0)
            transfer_pred.append(pred_class_0_1)
            transfer_pred.append(pred_class_2_1)
            transfer_pred.append(pred_class_0_2)
            transfer_pred.append(encoder_pred_class_2)
        else:
            latent_transfer_loss = 0

        if phase == 'Train':
            discri_fake_0 = self.discri_0_fc(recon_0_img)

            discri_fake_0_1 = self.discri_1_fc(recon_0_1_img)

            discri_fake_0_2 = self.discri_2_fc(recon_0_2_img)

            discri_real_0 = self.discri_0_fc(input_data_0)

            discri_fake_1 = self.discri_1_fc(recon_1_img)

            discri_fake_1_0 = self.discri_0_fc(recon_1_0_img)

            discri_fake_1_2 = self.discri_2_fc(recon_1_2_img)

            discri_real_1 = self.discri_1_fc(input_data_1)

            discri_fake_2 = self.discri_2_fc(recon_2_img)
 
            discri_fake_2_0 = self.discri_0_fc(recon_2_0_img)

            discri_fake_2_1 = self.discri_1_fc(recon_2_1_img)

            discri_real_2 = self.discri_2_fc(input_data_2)

            result.append(pred_class_0)
            result.append(pred_class_1)
            result.append(pred_class_2)
            result.append(code_0)
            result.append(latent_z_0)
            result.append(latent_z_1)
            result.append(latent_z_2)
            result.append(latent_loss)
            result.append(latent_loss)
            result.append(recon_0_img)
            result.append(recon_1_img)
            result.append(recon_2_img)
            result.append(discri_fake_0)
            result.append(discri_fake_1)
            result.append(discri_fake_2)
            result.append(discri_real_0)
            result.append(discri_real_1)
            result.append(discri_real_2)
            result.append(recon_1_0_img)
            result.append(recon_2_0_img)
            result.append(recon_0_1_img)
            result.append(recon_2_1_img)
            result.append(recon_0_2_img)
            result.append(recon_1_2_img)
            result.append(mean_loss)
            result.append(discri_fake_0_1)
            result.append(discri_fake_0_2)
            result.append(discri_fake_1_0)
            result.append(discri_fake_1_2)
            result.append(discri_fake_2_0)
            result.append(discri_fake_2_1)
        else:
            result.append(pred_class)
#        result.append(recon_img)
        return result, transfer_pred
    def netd(self, inter_data_0=0,inter_data_1=0,inter_data_2=0):
        result = []
        discri_0_inter_code = self.discri_0_conv(inter_data_0)
        discri_0_inter_code = discri_0_inter_code.view(-1, 128 * 6 * 6)
        discri_0_inter_feat_1 = self.discri_0_fc_1(discri_0_inter_code)
        discri_0_inter_feat_2 = self.discri_0_fc_2(discri_0_inter_feat_1)
        discri_inter_0 = self.discri_0_fc_3(discri_0_inter_feat_2)

        gradients_0 = autograd.grad(outputs=discri_inter_0 , inputs=inter_data_0,
                              grad_outputs=torch.ones(discri_inter_0.size()).cuda() ,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty_0 = ((gradients_0.norm(2, dim=1) - 1) ** 2).mean() * 10

        discri_1_inter_code = self.discri_1_conv(inter_data_1)
        discri_1_inter_code = discri_1_inter_code.view(-1, 128 * 6 * 6)
        discri_1_inter_feat_1 = self.discri_1_fc_1(discri_1_inter_code)
        discri_1_inter_feat_2 = self.discri_1_fc_2(discri_1_inter_feat_1)
        discri_inter_1 = self.discri_1_fc_3(discri_1_inter_feat_2)

        gradients_1 = autograd.grad(outputs=discri_inter_1, inputs=inter_data_1,
                              grad_outputs=torch.ones(discri_inter_1.size()).cuda() ,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty_1 = ((gradients_1.norm(2, dim=1) - 1) ** 2).mean() * 10

        discri_2_inter_code = self.discri_2_conv(inter_data_2)
        discri_2_inter_code = discri_2_inter_code.view(-1, 128 * 6 * 6)
        discri_2_inter_feat_1 = self.discri_2_fc_1(discri_2_inter_code)
        discri_2_inter_feat_2 = self.discri_2_fc_2(discri_2_inter_feat_1)
        discri_inter_2 = self.discri_2_fc_3(discri_2_inter_feat_2)

        gradients_2 = autograd.grad(outputs=discri_inter_2 , inputs=inter_data_2,
                              grad_outputs=torch.ones(discri_inter_2.size()).cuda() ,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty_2 = ((gradients_2.norm(2, dim=1) - 1) ** 2).mean() * 10
        #pdb.set_trace()
        result.append(gradient_penalty_0)
        result.append(gradient_penalty_1)
        result.append(gradient_penalty_2)
        return result
