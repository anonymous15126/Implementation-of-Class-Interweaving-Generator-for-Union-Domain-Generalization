import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torch.autograd import Variable
from torchvision import transforms
#from data_loader import GetLoader
from torchvision import datasets
from model_compat import DSN
import torchvision.utils as vutils
import pdb
def test(epoch, name, test_data , test_class_label, rec_mode, model_root,my_net):

    ###################
    # params          #
    ###################
    cuda = True
    cudnn.benchmark = True
    batch_size = 32
    image_size = 28
    #rec_mode = 'private'
    ###################
    # load data       #
    ###################

    #pdb.set_trace()
    i = 0
    n_total = 0
    n_correct = 0
    numbatch = int(test_data.shape[0]/batch_size)
    while i < numbatch:
        if i < numbatch - 1:
            img = test_data[i*batch_size:(i + 1) * batch_size,:]
            label = test_class_label[i * batch_size : (i+1) * batch_size]
        else:
            img = test_data[i*batch_size:,:]
            label = test_class_label[i * batch_size : ]

        img = torch.from_numpy(img)
        label = torch.from_numpy(label)
        #data_input = data_iter.next()
        #img, label = data_input

        batch_size = len(label)

        input_img = torch.FloatTensor(batch_size, 1, image_size, image_size)
        class_label = torch.LongTensor(batch_size)
        label = label.long()
        if cuda:
            img = img.cuda()
            label = label.cuda()
            input_img = input_img.cuda()
            class_label = class_label.cuda()

        #input_img.resize_as_(input_img).copy_(img)
        #class_label.resize_as_(label).copy_(label)
        inputv_img = Variable(img)
        classv_label = Variable(label)
        result_temp,temp = my_net(input_data_0 = inputv_img, input_data_1 = inputv_img,input_data_2 = inputv_img,phase = 'Test')
        pred = result_temp[0].data.max(1,keepdim=True)[1]
        #result_1 = my_net(input_data = inputv_img)
        #result_2 = my_net(input_data = inputv_img)
        #result_temp = result_0[0].data + result_1[0].data + result_2[0].data
        #pred = result_temp.max(1,keepdim=True)[1]
#        pred = result[0].data.max(1,keepdim=True)[1]
        n_correct += pred.eq(classv_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct * 1.0 / n_total
    #print(n_total,i)
    print('epoch: %d, accuracy of the %s dataset: %f' % (epoch, name, accu))
