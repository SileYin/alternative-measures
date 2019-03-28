import argparse
import os
import numpy as np

import torch
from torch.autograd import Variable
import torch.utils.data
import torch.nn as nn
from utils import prepare_datasets, evaluate,save,load,MyDataset
from models import Artist, Genre, Key
import torch.optim as optim
from tqdm import tqdm
import pickle

# Training settings
parser = argparse.ArgumentParser(description='8903 CNN project')
# Hyperparameters
parser.add_argument('--lr', type=float, metavar='LR', default=0.001,
                    help='learning rate')
# parser.add_argument('--momentum', type=float, metavar='M',
#                     help='SGD momentum')
parser.add_argument('--weight-decay', type=float, default=0.0,
                    help='Weight decay hyperparameter')
parser.add_argument('--batch-size', type=int, metavar='N', default=32,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, metavar='N', default=50,
                    help='number of epochs to train')
parser.add_argument('--model', default='key',
                    choices=['artist', 'genre', 'key'],
                    help='which model to train/evaluate')
parser.add_argument('--save-dir', default='models/')
# Other configuration
parser.add_argument('--no-cuda', action='store_true', default=True,
                    help='disables CUDA training')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(0)
np.random.seed(0)
if args.cuda:
    torch.cuda.manual_seed(0)

############# fetch torch Datasets ###################
######### you may change the dataset split % #########
train_set, val_set, test_set = '', '', ''
if args.model == 'artist':
    train_set, val_set, test_set = prepare_datasets(
                                                    chroma_path='./data/chroma_train_artist.npy',
                                                    label_path='./data/label_train_artist.npy',
                                                    splits=[0.7, 0.15, 0.15])
elif args.model == 'genre':
    train_set, val_set, test_set = prepare_datasets(
                                                    chroma_path='./data/chroma_train_genre.npy',
                                                    label_path='./data/label_train_genre.npy',
                                                    splits=[0.7, 0.15, 0.15])
elif args.model == 'key':
    train_set, val_set, test_set = prepare_datasets(
                                                    chroma_path='./data/chroma_train_key.npy',
                                                    label_path='./data/label_train_key.npy',
                                                    splits=[0.7, 0.15, 0.15])
else:
    raise Exception('Incorrect model name')
#prepare_datasets()

#print("XXXX  ",train_set[0])

############# create torch DataLoaders ###############
########### you may change the batch size ############
#print("a ",train_set[0][0][:][:],"b ",train_set[0][0][1][:])
#print("i",len(train_set),"j ",len(list(train_set[:])),"k ",len(list(train_set[:][:])))
# tlist = []
# clist = []
# labs = []
# print(len(train_set))
# for i in range(len(train_set)):
#     timbres = []
#     for k in range(len(train_set[i][0])):
#         timbres.append(train_set[i][0][k])
#     tlist.append(timbres)
#     chromas = []
#     for k in range(len(train_set[i][1])):
#         chromas.append(train_set[i][1][k])
#     clist.append(chromas)
#     for k in range(len(train_set[i][2])):
#         labs.append(train_set[i][2][k])

# tlist_v = []
# clist_v = []
# labs_v = []
# print("train val labs ",len(val_set[2:]))

# for i in range(len(val_set)-1):
# #print('dim 2 ',len(train_set[i][0]))
# #print('t ',len(timbres))
#     timbres = []
#     for k in range(len(val_set[i][0])):
#         #print("k ",len(train_set[i][j][k]))
#         #print("t ",train_set[i][j][k][:],"c ",train_set[i][j][k][1:])
#         timbres.append(val_set[i][0][k])
#     tlist_v.append(timbres)
#     #print(len(tlist[i][1]))
#     chromas = []
#     for k in range(len(val_set[i][1])):
#         chromas.append(val_set[i][1][k])
#     clist_v.append(chromas)
#     #print("mmmm",len(val_set[i][2]))
#     for k in range(len(val_set[i][2])):
#         labs_v.append(val_set[i][2][k])
# #print("xxx ",len(labs_v))
# #print("yyyy",len(clist_v))

# val_list = (tlist_v,clist_v,labs_v)







# train_t_loader = torch.utils.data.DataLoader(timbres, batch_size = args.batch_size, shuffle=True)
# train_c_loader = torch.utils.data.DataLoader(chromas, batch_size = args.batch_size, shuffle=True)
# val_loader = torch.utils.data.DataLoader(val_set, batch_size = 32)
# test_loader = torch.utils.data.DataLoader(test_set, batch_size = 32)
# print('loader : ', train_loader)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=1)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1)
print('loader : ', train_loader)

################ initialize the model ################
if args.model == 'artist':
    model = Artist()
elif args.model == 'genre':
    model = Genre()
elif args.model == 'key':
    model = Key()
else:
    raise Exception('Incorrect model name')

if args.cuda:
    model.cuda()
fp = open('data/chorddict.json','rb')
chorddict = pickle.load(fp)
fp.close()

######## Define loss function and optimizer ##########
############## Write your code here ##################
#Loss function
#model = model.float()
loss = torch.nn.CrossEntropyLoss()
#model.to(torch.float64)   
#Optimizer
optimizer = optim.Adam(model.parameters(), lr=args.lr)


######################################################


def train(epoch):
    """ Runs training for 1 epoch
    epoch: int, denotes the epoch number for printing
    """
    ############# Write train function ###############
    mean_training_loss = 0.0
    running_loss = 0.0
    lambda2 = lambda epoch: 0.95 ** epoch
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda2)


    model.train()

    correct = 0.0
    running_examples = 0.0
    eval_matrix = []

    for i,batch in tqdm(enumerate(train_loader)):
        inputs_a, inputs_b,labels = batch
        #print('labs ',len(labels))
        #print('l ',labels)
        tgts = []
        for x in range(len(labels)):
            lab = Variable(labels[x].to(torch.long))
            tgts.append(lab)
        for x in range(len(tgts)):
            model.init_hidden(1)
            optimizer.zero_grad()
            #print('i a ',inputs_a[x])
            outputs = model(inputs_a[x].to(torch.float),inputs_b[x].to(torch.float))
            #print("o ",outputs[0][0].unsqueeze(0))
            loss_size = loss(outputs[0][0].unsqueeze(0), tgts[x])
            #print("ls ",loss_size)
            loss_size.backward()
            optimizer.step()
            #print("ods ",outputs.data[0][:][0:5])

            _, predicted = torch.max(outputs[0][0].unsqueeze(0), 1)
            #print("predicted ",predicted)
            labels = tgts[x].view(-1)
            #print(predicted,labels)
            running_loss += loss_size.item()
            if np.isnan(running_loss):
                print("i ",inputs_b)
                print("o ",outputs[0][0].unsqueeze(0))
            #print("rl ",running_loss)
            # if i > 600:
            #     if x > len(tgts) * 0.9:
            #         print("input ",inputs_b[x])
            #         print("output ", outputs[0][0])
            #         print("predicted ",predicted)
            #         print("label ",labels)
            #         print("correct ",correct)
            
            correct += ((predicted.to(torch.long) == tgts[x].view(-1)).sum().item()) * inputs_b[x][:,0].sum().item()
            pred_decoded = list(chorddict.keys())[list(chorddict.values()).index(predicted.item())]
            cor_decoded = list(chorddict.keys())[list(chorddict.values()).index(tgts[x].view(-1))]
            eval_matrix.append((i,pred_decoded,cor_decoded))

            # if (predicted == tgts[x].view(-1)).sum().item() > 0:
            #     print("pred ",predicted,"tgt ",tgts[x].view(-1),"cor ",correct)
            #     print("ods ",outputs.data[0][0][0:5])
            #     print("x", x)
            running_examples += 1 * inputs_b[x][:,0].sum().item()
           
        
    mean_training_loss = running_loss/running_examples
    accuracy =  correct / running_examples
    print('Training Epoch:', epoch)
    print('Training Loss: {:.6f} \t'
            'Training Acc.: {:.6f}'.format(
            mean_training_loss, accuracy))
    fname = 'data/' + str(epoch) + '.log'
    
    fp = open(fname,'wb')
    pickle.dump(eval_matrix,fp)
    fp.close()
    ##################################################
    return 
def adjust_learning_rate(optimizer, epoch, adjust_every):
    """
    Adjusts the learning rate of the optimizer based on the epoch
    Args:
       optimizer:      object, of torch.optim class 
       epoch:          int, epoch number
       adjust_every:   int, number of epochs after which adjustment is to done
    """
    #######################################
    ### BEGIN YOUR CODE HERE
    # write your code to adjust the learning
    # rate based on the epoch number
    # You are free to implement your own 
    # method here
    #######################################

    if epoch % adjust_every == (adjust_every - 1):
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.5

    #######################################
    ### END OF YOUR CODE
    #######################################
    return

######## Training and evaluation loop ################
######## Save model with best val accuracy  ##########

for i in range(args.epochs):
    # if i > 43:
    #     break
    # batch_offset = 0
    # batch_offset = train(i,batch_offset)
    #train_set,val_set,_ = shuffle(splits=[0.7, 0.15, 0.15])
    train(i)
    criterion = loss
    #val_loss, val_acc = evaluate(val_list,args.batch_size, model, criterion, args.cuda)
    val_loss, val_acc = evaluate(val_loader, model, criterion, args.cuda)
    adjust_learning_rate(optimizer, i, 8)
    #print('loss :', val_loss, 'accuracy :', val_acc)
    print('Validation Loss: {:.6f} \t'
            'Validation Acc.: {:.6f}'.format(
            val_loss, val_acc))
    ####### write saving code here ###################
    if args.model == 'artist':
        save(model, 'Artist.pth')
    elif args.model == 'genre':
        save(model, 'Genre.pth')
    elif args.model == 'key':
        save(model, 'Key.pth')


############ write testing code here #################
def test(model):
    test_loss = 0.0
    mean_training_loss = 0.0
    model.eval()
    correct = 0
    running_examples = 0
    for i, batch in enumerate(test_loader):
        ############ Write your code here ############
        #Get inputs
        inputs_a, inputs_b, labels = batch
        # labels = torch.LongTensor(np.array(labs[offset:offset + args.batch_size]).astype(int))
        # labels = labels.reshape(args.batch_size,1)
        labels = labels.type(torch.LongTensor) 
        model.init_hidden(inputs_a.shape[0])         
        #inputs = np.reshape(inputs,(len(inputs),1,128,130))
        #Forward pass, backward pass, optimize
        outputs = model(inputs_a,inputs_b)
        loss_size = loss(outputs.data, labels.view(-1))
          
        #Print statistics
        test_loss += loss_size.item()
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels.view(-1)).sum().item()
        running_examples += len(labels)
        # print('correct :',correct,'predicted :',predicted)
    test_acc =  correct / running_examples

    print('Test Loss: {:.6f} \t'
            'Test Acc.: {:.6f}'.format(
            test_loss, test_acc))

############# Load best model and test ###############
if args.model == 'artist':
    msd = load('artist.pth')
    model.load_state_dict(msd)
elif args.model == 'genre':
    msd = load('genre.pth')
    model.load_state_dict(msd)
elif args.model == 'key':
    msd = load('key.pth')
    model.load_state_dict(msd)

test(model)
############## Write your code here ##################
