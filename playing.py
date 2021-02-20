import logging
#On local mac use base conda env and Python command

#from models.word_model import RNNModel
#from text_helper import TextHelper


#TODO
#Read through code line by line and decide what is being useful and what not
#add doc strings

logger = logging.getLogger('logger')

import json
from datetime import datetime
import argparse
from scipy import ndimage
import torch
import torchvision
import os
import torchvision.transforms as transforms
from collections import defaultdict, OrderedDict
from tensorboardX import SummaryWriter
import torchvision.models as models
#from models.mobilenet import MobileNetV2
from helper import Helper
from image_helper import ImageHelper
#from models.densenet import DenseNet
#from models.simple import Net, FlexiNet, reseed
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm as tqdm
import time
import random
import yaml
from utils.text_load import *
from models.resnet import Res, PretrainedRes
from utils.utils import dict_html, create_table, plot_confusion_matrix
from inception import *



#Added after the fact
from multiprocessing import freeze_support
freeze_support()

# Add wandb logging which is synced with the Tensorboard
import wandb 
#wandb.init(sync_tensorboard=True)



#Setting up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.is_available():
    print("OMG cuda is available")

layout = {'cosine': {
    'cosine': ['Multiline', ['cosine/0',
                                         'cosine/1',
                                         'cosine/2',
                                         'cosine/3',
                                         'cosine/4',
                                         'cosine/5',
                                         'cosine/6',
                                         'cosine/7',
                                         'cosine/8',
                                         'cosine/9']]}}


def plot(x, y, name):
    """
    Add to the writer
    """
    freeze_support()
    writer.add_scalar(tag=name, scalar_value=y, global_step=x)


def compute_norm(model, norm_type=2):
    """
    Normalize data
    TODO check if correct
    """
    freeze_support()
    total_norm = 0
    for p in model.parameters():
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm



def test(net, epoch, testloader, vis=True):
    """
    TBD
    net: neural network:
    epoch: int: no of epochs
    testloader: DataLoader: TODO what form exactly
    vis: boolean: If true print out tables and accuracies for all subgroups

    return: float: overall model accuracy
    """
    freeze_support()
    #Sets module in evaluation mode
    net.eval()
    correct = 0
    total = 0
    i = 0
    correct_labels = []
    predict_labels = []
    
    female_total = 0
    female_correct = 0
    female_correct_labels = []
    female_predict_labels = []
    
    male_total = 0
    male_correct = 0
    male_correct_labels = []
    male_predict_labels = []


    #with statement ensures proper aquicition and release of resources
    #no_grad() is a context manager that disables gradient calculation
    with torch.no_grad():
        #Tqdm prints progress bar
        #TODO look into hwo testloader works
        #so we are loading all the data then predict for it and evaluate
        for data in tqdm(testloader):

            inputs, protected_labels, labels = data #Setting up Data
            #Protected labels are the labels of the subgroups whose performance we compare


            inputs = inputs.to(device) #device is process
            labels = labels.to(device)

            outputs = net(inputs) #Get outputs from model

            #returns the max of all input
            _, predicted = torch.max(outputs.data, 1)

            predict_labels.extend([x.item() for x in predicted]) #.item returns value of tensor in standard python value
            correct_labels.extend([x.item() for x in labels])

            #getting total items
            total += labels.size(0)
            #Counting correct items
            correct += (predicted == labels).sum().item()
            
            #Sort predictions and correct labels according to protected label
            for count, i in enumerate(protected_labels):
                if i == 0:
                    female_predict_labels.append(predicted[count].item())
                    female_correct_labels.append(labels[count].item())
                else:
                    male_predict_labels.append(predicted[count].item())
                    male_correct_labels.append(labels[count].item())
            
        
        #Get the accuracy for both female and male subsets
        for predicted_label, correct_label in zip(female_predict_labels, female_correct_labels):
            female_total += 1
            if predicted_label == correct_label:
                female_correct += 1

        for predicted_label, correct_label in zip(male_predict_labels, male_correct_labels):
            male_total += 1
            if predicted_label == correct_label:
                male_correct += 1
    

    #Logger which is writing to terminal and to wandb if activated
    logger.info(f'Epoch {epoch}. acc: {100 * correct / total}. total values: {total}')
    logger.info(f'Epoch {epoch}. female acc: {100 * female_correct / female_total} total values: {female_total}')
    if male_total > 0:
        logger.info(f'Epoch {epoch}. male acc: {100 * male_correct / male_total} total values: {male_total}')

    main_acc = 100 * correct / total #Always gets overridden.
    

    i = 0

    if vis:
        #Plot important info for main performance

        #Plot UN-nomralized confusion matrix
        main_fig, main_cm = plot_confusion_matrix(correct_labels,
                                                  predict_labels,
                                                  labels=helper.labels,
                                                  normalize=False)
        
        #Add matrix to writer
        writer.add_figure(figure=main_fig, global_step=epoch, tag='tag/unnormalized_cm')

        # Accuracy
        plot(epoch, 100 * correct / total, "Main Accuracy")

        # Precision
        precision = (main_cm[1][1]/(main_cm[1][1] + main_cm[0][1]))
        plot(epoch, precision, "Main Precision")

        # Recall 
        recall = (main_cm[1][1]/(main_cm[1][1] + main_cm[1][0]))
        plot(epoch, recall, "Main Recall")

        # F1-Score
        f1_score = ((2 * precision * recall) / (precision + recall))
        plot(epoch, f1_score, "Main F1")
        



        #Plot important iinfo for Female performance
        plot(epoch, 100 * female_correct / female_total, "Female Accuracy")
        female_fig, female_cm = plot_confusion_matrix(female_correct_labels,
                                                      female_predict_labels,
                                                      labels=helper.labels,
                                                      normalize=False)
        writer.add_figure(figure=female_fig, global_step=epoch, tag='tag/female_unnormalized_cm')
        # Accuracy
        plot(epoch, 100 * female_correct / female_total, "Female Accuracy")
        # Precision
        female_precision = (female_cm[1][1]/(female_cm[1][1] + female_cm[0][1]))
        plot(epoch, female_precision, "Female Precision")
        # Recall 
        female_recall = (female_cm[1][1]/(female_cm[1][1] + female_cm[1][0]))
        plot(epoch, female_recall, "Female Recall")
        # F1-Score
        female_f1_score = ((2 * female_precision * female_recall) / (female_precision + female_recall))
        plot(epoch, female_f1_score, "Female F1")




        #Plot important info for Male performance if there are any males
        #TODO find out why they would use this?

        if male_total > 0:
            male_fig, male_cm = plot_confusion_matrix(male_correct_labels,
                                                      male_predict_labels,
                                                      labels=helper.labels,
                                                      normalize=False)
            writer.add_figure(figure=male_fig, global_step=epoch, tag='tag/male_unnormalized_cm')
            # Accuracy
            plot(epoch, 100 * male_correct / male_total, "Male Accuracy")
            # Precision
            male_precision = (male_cm[1][1]/(male_cm[1][1] + male_cm[0][1]))
            plot(epoch, male_precision, "Male Precision")
            # Recall 
            male_recall = (male_cm[1][1]/(male_cm[1][1] + male_cm[1][0]))
            plot(epoch, male_recall, "Male Recall")
            # F1-Score
            male_f1_score = ((2 * male_precision * male_recall) / (male_precision + male_recall))
            plot(epoch, male_f1_score, "Male F1")
    
    #Returning main accuracy
    return 100 * correct / total #Could just reuse the main keyword


def train_dp(trainloader, model, optimizer, epoch):
    """
    Train NN with differential privacy
    trainloader: DataLoader:
    model: e.g ResNet: 
    optimizer: adam/SGD
    epoch: int : the amount of epochs to be run
    """

    freeze_support()
    norm_type = 2 #TODO check if still needed not used
    #Set model in training mode
    model.train()

    running_loss = 0.0
    #
    label_norms = defaultdict(list)
    ssum = 0
    
    with tqdm(total=len(trainloader), leave=True) as pbar:
        for i, data in enumerate(trainloader, 0):
            inputs, protected_labels, labels = data #replace protected_labels
            #TODO inspect data to see what is actually going on
            #inputs, labels = data
            #protected_labels = labels

            inputs = inputs.to(device)
            labels = labels.to(device)
            #Clears gradient of all optimizers
            optimizer.zero_grad()


            outputs = model(inputs)
            #Cross entropy loss between the output of the model and the labels
            loss = criterion(outputs, labels)
            running_loss += torch.mean(loss).item()
            #Record losses in tensor form
            #tensor getting reshaped to microbatch and -1
            losses = torch.mean(loss.reshape(num_microbatches, -1), dim=1)

            saved_var = dict()

            #Set up saved var with place holders
            for tensor_name, tensor in model.named_parameters():
                saved_var[tensor_name] = torch.zeros_like(tensor)

            
            grad_vecs = dict() #Gradient vectors?

            count_vecs = defaultdict(int)

            #Iterate through the losses
            for pos, j in enumerate(losses):
                j.backward(retain_graph=True)

                #TODO find out what is going on here
                if helper.params.get('count_norm_cosine_per_batch', False):
                    grad_vec = helper.get_grad_vec(model, device)
                    label = labels[pos].item()
                    count_vecs[label] += 1
                    if grad_vecs.get(label, False) is not False:
                        grad_vecs[label].add_(grad_vec)
                    else:
                        grad_vecs[label] = grad_vec

                #Gradients are clipped by Amount S
                #TODO S never gets passed down to funciton bad coding practice
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), S)


                if helper.params['dataset'] == 'dif':#TODO this can be deleted as not being used
                    label_norms[f'{labels[pos]}_{helper.label_skin_list[idxs[pos]]}'].append(total_norm)
                else:
                    label_norms[int(labels[pos])].append(total_norm)#Add the clipped norms to label norms

                #.named_parameters returns an iterator over model parameters
                for tensor_name, tensor in model.named_parameters():
                      if tensor.grad is not None:
                        new_grad = tensor.grad
                    #logger.info('new grad: ', new_grad)
                        saved_var[tensor_name].add_(new_grad)
                #Sets all gradients of model to zero
                model.zero_grad()

            for tensor_name, tensor in model.named_parameters():
                if tensor.grad is not None:
                    #Add the gradient 
                    if device.type == 'cuda':
                        saved_var[tensor_name].add_(torch.cuda.FloatTensor(tensor.grad.shape).normal_(0, sigma))
                    else:
                        saved_var[tensor_name].add_(torch.FloatTensor(tensor.grad.shape).normal_(0, sigma))
                    #Setting tensor gradient to saved gradient over number of microbatches
                    tensor.grad = saved_var[tensor_name] / num_microbatches #Does this mean it adds a TODO
            
            #TODO not sure what is going on here might be how well the step size is doing
            if helper.params.get('count_norm_cosine_per_batch', False):
                total_grad_vec = helper.get_grad_vec(model, device)
                # logger.info(f'Total grad_vec: {torch.norm(total_grad_vec)}')
                for k, vec in sorted(grad_vecs.items(), key=lambda t: t[0]):
                    vec = vec/count_vecs[k]
                    cosine = torch.cosine_similarity(total_grad_vec, vec, dim=-1)
                    distance = torch.norm(total_grad_vec-vec)
                    # logger.info(f'for key {k}, len: {count_vecs[k]}: {cosine}, norm: {distance}')

                    plot(i + epoch*len(trainloader), cosine, name=f'cosine/{k}')
                    plot(i + epoch*len(trainloader), distance, name=f'distance/{k}')

            optimizer.step() #Make a step in direction

            if i > 0 and i % 20 == 0:
                #             logger.info('[%d, %5d] loss: %.3f' %
                #                   (epoch + 1, i + 1, running_loss / 2000))
                plot(epoch * len(trainloader) + i, running_loss, 'Train Loss')
                running_loss = 0.0
            #manually update the progess bar
            pbar.update(1)
            
    print(ssum)
    for pos, norms in sorted(label_norms.items(), key=lambda x: x[0]):
        logger.info(f"{pos}: {torch.mean(torch.stack(norms))}")
        if helper.params['dataset'] == 'dif':
            plot(epoch, torch.mean(torch.stack(norms)), f'dif_norms_class/{pos}')
        else:
            plot(epoch, torch.mean(torch.stack(norms)), f'norms/class_{pos}')


def train(trainloader, model, optimizer, epoch):
    freeze_support()
    model.train()
    running_loss = 0.0


    with tqdm(total=len(trainloader), leave=True) as pbar:
        for i, data in enumerate(trainloader, 0):
            inputs, protected_labels, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            # logger.info statistics
            running_loss += loss.item()
            if i > 0 and i % 20 == 0:
                #             logger.info('[%d, %5d] loss: %.3f' %
                #                   (epoch + 1, i + 1, running_loss / 2000))
                plot(epoch * len(trainloader) + i, running_loss, 'Train Loss')
                running_loss = 0.0
            pbar.update(1)


if __name__ == '__main__':
    freeze_support()
    parser = argparse.ArgumentParser(description='PPDL')
    parser.add_argument('--params', dest='params', default='utils/params_celeba.yaml')
    parser.add_argument('--name', dest='name', required=True)

    args = parser.parse_args()
    d = datetime.now().strftime('%b.%d_%H.%M.%S')
    writer = SummaryWriter(log_dir=f'runs/{args.name}')
    writer.add_custom_scalars(layout)
    

    with open(args.params) as f:
        params = yaml.load(f)
    if params.get('model', False) == 'word':
        print("RIP text helper")
    else:
        helper = ImageHelper(current_time=d, params=params, name=args.name)


    logger.addHandler(logging.FileHandler(filename=f'{helper.folder_path}/log.txt'))
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.DEBUG)
    logger.info(f'current path: {helper.folder_path}')

    # --- setting up parameters ---

    #Set batch size
    batch_size = int(helper.params['batch_size'])
    #Set num of microbatches
    num_microbatches = int(helper.params['num_microbatches'])
    lr = float(helper.params['lr'])

    momentum = float(helper.params['momentum'])

    decay = float(helper.params['decay'])

    epochs = int(helper.params['epochs'])

    S = float(helper.params['S'])

    z = float(helper.params['z'])

    #Calculate Sigma
    sigma = z * S
    dp = helper.params['dp']
    mu = helper.params['mu']

    logger.info(f'DP: {dp}')

    logger.info(batch_size)
    logger.info(lr)
    logger.info(momentum)

    #reseed(5)
    if helper.params['dataset'] == 'inat':
        helper.load_inat_data()
        #helper.balance_loaders() todo reactivate
    # elif helper.params['dataset'] == 'word':
    #     helper.load_data()
    # elif helper.params['dataset'] == 'dif':
    #     helper.load_dif_data()
    #     helper.get_unbalanced_faces()
    #THE ONLY CASE WE ARE USING
    elif helper.params['dataset'] == 'celeba':
        helper.load_celeba_data()
    #Nice to keep this if using other datasets
    else:
        helper.load_cifar_data(dataset=params['dataset'])
        logger.info('before loader')
        helper.create_loaders()
        logger.info('after loader')
        helper.sampler_per_class()
        logger.info('after sampler')
        helper.sampler_exponential_class(mu=mu, total_number=params['ds_size'], key_to_drop=params['key_to_drop'],
                                        number_of_entries=params['number_of_entries'])
        logger.info('after sampler expo')
        helper.sampler_exponential_class_test(mu=mu, key_to_drop=params['key_to_drop'],
              number_of_entries_test=params['number_of_entries_test'])
        logger.info('after sampler test')

    helper.compute_rdp() #Comput rdp ?TODO definintion residue dP


    #Set no of classes
    if helper.params['dataset'] == 'cifar10':
        num_classes = 10
    # elif helper.params['dataset'] == 'cifar100':
    #     num_classes = 100
    # elif helper.params['dataset'] == 'inat':
    #     num_classes = len(helper.labels)
    #     logger.info('num class: ', num_classes)  
    # elif helper.params['dataset'] == 'dif':
    #     num_classes = len(helper.labels)
    # --- Again only need this realisticly
    elif helper.params['dataset'] == 'celeba':
        num_classes = len(helper.labels)
    else:
        num_classes = 10



    #Set up the neural network


    #reseed(5)
    if helper.params['model'] == 'densenet':
        net = DenseNet(num_classes=num_classes, depth=helper.params['densenet_depth'])
    # --- Using this one only
    elif helper.params['model'] == 'resnet':
        logger.info(f'Model size: {num_classes}')
        net = models.resnet18(num_classes=num_classes)
    elif helper.params['model'] == 'PretrainedRes': #
        net = models.resnet18(pretrained=True)
        net.fc = nn.Linear(512, num_classes)
        net = net.cuda()
    elif helper.params['model'] == 'FlexiNet':
        net = FlexiNet(3, num_classes)
    elif helper.params['model'] == 'dif_inception':
        net = inception_v3(pretrained=True, dif=True)
        net.fc = nn.Linear(768, num_classes)
        net.aux_logits = False
    elif helper.params['model'] == 'inception':
        net = inception_v3(pretrained=True)
        net.fc = nn.Linear(2048, num_classes)
        net.aux_logits = False
        #model = torch.nn.DataParallel(model).cuda()
    elif helper.params['model'] == 'mobilenet':
        net = MobileNetV2(n_class=num_classes, input_size=64)
    elif helper.params['model'] == 'word':
        net = RNNModel(rnn_type='LSTM', ntoken=helper.n_tokens,
                 ninp=helper.params['emsize'], nhid=helper.params['nhid'],
                 nlayers=helper.params['nlayers'],
                 dropout=helper.params['dropout'], tie_weights=helper.params['tied'])
    else:
        net = Net()


    #GPU set-up
    if helper.params.get('multi_gpu', False):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        logger.info(f"Let's use {torch.cuda.device_count()} GPUs!")
        net = nn.DataParallel(net)

    net.to(device)


    #Resume model training
    if helper.params.get('resumed_model', False):
        logger.info('Resuming training...')
        loaded_params = torch.load(f"saved_models/{helper.params['resumed_model']}")
        net.load_state_dict(loaded_params['state_dict'])
        helper.start_epoch = loaded_params['epoch']
        # helper.params['lr'] = loaded_params.get('lr', helper.params['lr'])
        logger.info(f"Loaded parameters from saved model: LR is"
                    f" {helper.params['lr']} and current epoch is {helper.start_epoch}")
    else:
        helper.start_epoch = 1
    


    logger.info(f'Total number of params for model {helper.params["model"]}: {sum(p.numel() for p in net.parameters() if p.requires_grad)}')
    
    #TODO why?
    #Cross entpropy loss configuration
    if dp:
        criterion = nn.CrossEntropyLoss(reduction='none')
    else:
        criterion = nn.CrossEntropyLoss()


    #Set up optimizer
    if helper.params['optimizer'] == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
    elif helper.params['optimizer'] == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=decay)
    else:
        raise Exception('Specify `optimizer` in params.yaml.')

    
    #Set learning rate to specific changing over milestones
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     milestones=[0.5 * epochs,
                                                                 0.75 * epochs], gamma=0.1)

    #Show model parameters
    table = create_table(helper.params)
    writer.add_text('Model Params', table)
    logger.info(table)
    logger.info(helper.labels)
    epoch =0
    # acc = test(net, epoch, "accuracy", helper.test_loader, vis=True) #seems like there is no test loader in helper

    #Depending on if dp or not train model for each epoch and at the end save the accuaray.
    for epoch in range(helper.start_epoch, epochs):  # loop over the dataset multiple times
        if dp:
            train_dp(helper.train_loader, net, optimizer, epoch)
        else:
            train(helper.train_loader, net, optimizer, epoch)
        if helper.params['scheduler']:
            scheduler.step()
        main_acc = test(net, epoch, helper.test_loader, vis=True)
        
        helper.save_model(net, epoch, main_acc)
    logger.info(f"Finished training for model: {helper.current_time}. Folder: {helper.folder_path}")




#GOOD behavior of reading someones code.. just read from the top down looking up each owrk making comments for ones own...
# Batch = evaluates all examples in set and then updates model -> that is why we divide the gradient
#ONe batch per epoch . Multiple data

#Note: we are using cross entropy loss
#NOte: TODO there is an option to switch to pretrained resnet should we be doing that? Is that what the paper did?


#Should there be a resume model option to be used in the paramters
#TODO: research
#what cross entropy and why are we using cross entropy reduction





#On the docker file which vast.ai is do:
# tqdm
# pytorch
# torchvision
# pyyaml
# tensorboardX
# scikit-learn
# matplotlib

# All i had to do was pip install these
# matplotlib
# scipy
# tensorboardX
# sklearn

# apt-get update
# apt-get -y install curl
#had to install wget
#and absl-py





#v
#Now we need to figure out how to mount the data properly
#https://we.tl/t-HuPJj6RbEG
#test file: https://we.tl/t-7SDA7EXiYg
#find a way to automate/test this..
#Idea:
# save it somewhere like in that file storage .
# make script to create file
# change files always locally upload for now
# then download/mount the file


#later make sure wandb works that would be a nice way to get back results


#curl 'https://download.wetransfer.com//eu2/6a0f7ee3c2742f01469ebe3a0c4dce8720210220153552/a822e8b2309b78b5224feea450df401597a7b9f2/s1-2.pdf?cf=y&token=eyJhbGciOiJIUzI1NiJ9.eyJleHAiOjE2MTM4Mzk2NTQsInVuaXF1ZSI6IjZhMGY3ZWUzYzI3NDJmMDE0NjllYmUzYTBjNGRjZTg3MjAyMTAyMjAxNTM1NTIiLCJmaWxlbmFtZSI6InMxLTIucGRmIiwid2F5YmlsbF91cmwiOiJodHRwOi8vcHJvZHVjdGlvbi5iYWNrZW5kLnNlcnZpY2UuZXUtd2VzdC0xLnd0OjkyOTIvd2F5YmlsbC92MS9zYXJrYXIvOTdhNDRmOGIxNTcyZDI1ZWY2NjRlYmYwNDAyM2YxZGQxYTdlNjE1ZDlmZTFkMmMzZDM1MWZkY2I1YmNhNzFiMzVjOGUzNjA5M2NhYWM3ODgwZDdhYmIiLCJmaW5nZXJwcmludCI6ImE4MjJlOGIyMzA5Yjc4YjUyMjRmZWVhNDUwZGY0MDE1OTdhN2I5ZjIiLCJjYWxsYmFjayI6IntcImZvcm1kYXRhXCI6e1wiYWN0aW9uXCI6XCJodHRwOi8vcHJvZHVjdGlvbi5mcm9udGVuZC5zZXJ2aWNlLmV1LXdlc3QtMS53dDozMDAwL3dlYmhvb2tzL2JhY2tlbmRcIn0sXCJmb3JtXCI6e1widHJhbnNmZXJfaWRcIjpcIjZhMGY3ZWUzYzI3NDJmMDE0NjllYmUzYTBjNGRjZTg3MjAyMTAyMjAxNTM1NTJcIixcImRvd25sb2FkX2lkXCI6MTE0OTczNDk1MDl9fSJ9.j0k0tMpmWtUemUtzij6SadpaXz8SVG5Q4-wok8MIK8U' --location --output my.pdf

#curl ''
#could be that it is better
#should try that curl comman to maybe save time

#Somehow not using gpu? why?








#python version is 379 dunno if that is cause its local or why...
#