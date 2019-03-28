import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from tqdm import tqdm
import pickle

torch.manual_seed(0)
np.random.seed(0)


class MyDataset(Dataset):
    def __init__(self, t_train, c_train, l_train):
        super(MyDataset, self).__init__()
        self.t_train = t_train
        self.c_train = c_train
        self.l_train = l_train

    def __getitem__(self, index):
        return self.t_train[index], self.c_train[index], self.l_train[index]

    def __len__(self):
        return len(self.l_train)


def prepare_datasets(timbre_path='./data/timbre.dat', chroma_path='./data/pitch.dat',
                     label_path='./data/label_train_key.npy', splits=[0.7, 0.15, 0.15]):
    ######################################################
    ################### Preparing Data ###################
    # timbre_path: 	Path to timbre vector
    # chroma_path: 	Path to pitch chroma vector
    # label_path:   Path to song labels (Track ID, artist, genre, and key)
    # splits: 		list of split percentages for dataset
    ######################################################

    assert np.sum(splits) == 1
    assert splits[0] != 0
    assert splits[1] != 0
    assert splits[2] != 0

    ########### load data into torch Tensors #############
    fp = open('data/pitch.dat','rb')
    pitches = pickle.load(fp)
    fp.close()
    print("len all",len(pitches))
    labs = []
    
    #print(labmatrix)
    songlist = []
    lablist = []
    labmatrix = []
    for h in range(len(pitches)):
        seglist = []
        #eventvector = np.asarray(pitches[h][1][0][0][2:14])
        labs = []
        for i in range(0,len(pitches[h][1])):
            #print("p ",pitches[0][1][i])

            if float(pitches[h][1][i][1][0]) == 0 and float(pitches[h][1][i][1][1]) == 0:
                #print("inner ",pitches[h][1][i][0]," ",pitches[h][1][i][1])
                eventlist = np.append(eventlist,np.asarray(pitches[h][1][i][0][2:14]))
                #eventlist = np.append(eventlist,np.argmax(pitches[h][1][i][0]))
                #print("pi ",pitchmat)
            else:
                if i == 0:
                    pass
                else:
                    seglist.append(eventlist)
                    #songlist.append(seglist)
                    #print("h ",h,"len ",len(pitches[h][1]),"e ",len(eventlist),"l ",len(labs))
                eventlist = np.empty(12)
                #eventlist = []
                #print("start ",pitchouter.shape," ",pitchmat.shape)
                eventlist = np.append(eventlist,np.asarray(pitches[h][1][i][0][2:14]))
                #eventlist = np.append(eventlist,np.argmax(pitches[h][1][i][0]))
                #print("p ",pitchmat)
                labs.append(pitches[h][1][i][1][2])
        seglist.append(eventlist)
        songlist.append(seglist)
        print("labs ",len(labs),"t ",type(labs)," ",len(seglist),"ts ",type(seglist))
        labmatrix.append(labs)
        print("h ",h,'song ',len(songlist),'lab ',len(labmatrix))
    #print("s ",len(songlist[888]),"l ",labmatrix[-1])
    for i in range(len(songlist)):
        print("sls ",len(songlist[i]))
    
                
    
        
    chroma_train = np.asarray(songlist)
    
    
    # for i inange(len(pitchouter)):
    #     print("i ",i," ",pitchouter.shape)
    fp = open('data/timbre.dat','rb')
    timbres = pickle.load(fp)
    fp.close()

    timbrefinal = []
    songlist = []
    for h in range(len(timbres)):
        seglist = []
        
        #print("len ",len(timbres[h][1]))
        for i in range(0,len(timbres[h][1])):
            #print("p ",pitches[0][1][i])
            if float(timbres[h][1][i][1][0]) == 0 and float(timbres[h][1][i][1][1]) == 0:
                #print("inner ",pitches[0][1][i][0]," ",pitches[0][1][i][1])
                eventlist = np.append(eventlist,np.asarray(timbres[h][1][i][0][2:14]))
                #print("pi ",pitchmat)
            else:
                if i == 0:
                    pass
                else:
                    seglist.append(eventlist)
                    #songlist.append(seglist)
                eventlist = np.empty(12)
                #print("start ",pitchouter.shape)
                eventlist = np.append(eventlist,np.asarray(timbres[h][1][i][0][2:14]))
                #print("p ",pitchmat)
        seglist.append(eventlist)
        songlist.append(seglist)
        #print('song ',len(songlist))
   
    timbre_train = np.asarray(songlist)
    #print('labmatrix ',labmatrix)
    label_train = np.asarray(labmatrix)
    
        
    X = MyDataset(timbre_train, chroma_train, label_train)

    ###### split dataset into training validation ########
    ###### and test. 0.7, 0.15, 0.15 split        ########

    
    n_points = label_train.shape[0]
    print("pts ",n_points)

    train_split = (int(splits[0] * n_points))
    val_split = (int(splits[1] * n_points))
    test_split = (int(splits[2] * n_points))

    shuffle_indices = np.random.permutation(np.arange(n_points))

    train_indices = shuffle_indices[0:train_split]
    print("tr ",train_indices.shape)
    val_indices = shuffle_indices[train_split:train_split + val_split]
    print("v ",val_indices.shape)
    test_indices = shuffle_indices[val_split:val_split + test_split]
    print("t ",test_indices.shape)
    ############# create torch Datasets ##################
    # train_set = torch.utils.data.TensorDataset(torch.Tensor(X[train_indices][0]),torch.Tensor(X[train_indices][1]), torch.Tensor(X[train_indices][2]))
    # val_set = torch.utils.data.TensorDataset(torch.Tensor(X[val_indices][0]),torch.Tensor(X[val_indices][1]), torch.Tensor(X[val_indices][2]))
    # test_set = torch.utils.data.TensorDataset(torch.Tensor(X[test_indices][0]),torch.Tensor(X[test_indices][1]), torch.Tensor(X[test_indices][2]))
    train_set = []
    timbreobs = []
    #print("x ",X[train_indices][0].shape)
    # chromaobs = X[train_indices][0].reshape(-1,X[train_indices][0].shape[0])
    # timbreobs = X[train_indices][1].reshape(-1,X[train_indices][1].shape[0])
    # labs = X[train_indices][2].reshape(-1,X[train_indices][2].shape[0])

    # print("after ",chromaobs.shape)
    for tsample in (X[train_indices][0]):
        timbreobs.append(tsample)
    chromaobs = []
    for csample in X[train_indices][1]:
        chromaobs.append(csample)
    labs = []
    for lsample in X[train_indices][2]:
        #lsamp3 = np.concatenate([lsample,np.array([0])])
        labs.append(lsample)
    x = zip(timbreobs,chromaobs,labs)
    for l in x:
        train_set.append(list(l))
        #print("t ",len(l[0]),"c ",len(l[1]),"l ",len(l[2]))
    print("tset len XXXXXXXXXXXX ",len(train_set))
   
    
    val_set = []
    timbreobs = []
    for tsample in (X[val_indices][0]):
        timbreobs.append(tsample)
    chromaobs = []
    for csample in X[val_indices][1]:
        chromaobs.append(csample)
    labs = []
    for lsample in X[val_indices][2]:
        #print("z")
        labs.append(lsample)
    #print("qqqq",len(labs))
    x = zip(timbreobs,chromaobs,labs)
    for l in x:
        val_set.append(list(l))
    #print("val labs",len(val_set[2:]))
    
    

    test_set = []
    timbreobs = []
    for tsample in (X[test_indices][0]):
        timbreobs.append(tsample)
    chromaobs = []
    for csample in X[test_indices][1]:
        chromaobs.append(csample)
    labs = []
    for lsample in X[test_indices][2]:
        labs.append(lsample)
    test_set = [timbreobs,chromaobs]
    x = zip(timbreobs,chromaobs,labs)
    for l in x:
        test_set.append(list(l))

    print("val set final ",len(val_set[2:]))
    return train_set, val_set, test_set


def evaluate(data_loader, model, criterion, cuda):
    ######################################################
    ################### Evaluate Model ###################
    # data_loader: 	pytorch dataloader for eval data
    # model: 		pytorch model to be evaluated
    # criterion: 	loss function used to compute loss
    # cuda:			boolean for whether to use gpu

    # Returns loss and accuracy
    ######################################################
    ######## WRITE YOUR CODE BELOW #######################
    mean_training_loss = 0.0
    running_loss = 0.0
    model.eval()
    correct = 0
    running_examples = 0
    
    for i, batch in tqdm(enumerate(data_loader)):
        
        inputs_a, inputs_b, labels = batch
        tgts = []
        for x in range(len(labels)):
            lab = Variable(labels[x])
            tgts.append(lab)
        for x in range(len(tgts)):
            model.init_hidden(1)
            outputs = model(inputs_a[x].to(torch.float),inputs_b[x].to(torch.float))
            loss_size = criterion(outputs[0][0].unsqueeze(0), tgts[x])
            _, predicted = torch.max(outputs[0][0].unsqueeze(0), 1)
            labels = tgts[x].view(-1)
            running_loss += loss_size.item()
            correct += (predicted == tgts[x].view(-1)).sum().item()
            print("pred ",predicted,"lba ",labels,"cor ",correct)
            running_examples += 1        
    accuracy =  correct / running_examples
    running_loss = running_loss / running_examples
    return running_loss, accuracy


def save(model, path):
    ######################################################
    ################### Save Model ###################
    # model: 	pytorch model to be saved
    # path:		path for model to be saved
    ######################################################
    ######## WRITE YOUR CODE BELOW #######################
    torch.save(model.state_dict(), path)

def load(path):
    ######################################################
    ################### Load Model ###################
    # path:		path of model to be loaded

    # Returns model state_dict
    ######################################################
    ######## WRITE YOUR CODE BELOW #######################
    state_dict = torch.load(path)

    ######################################################
    return state_dict
