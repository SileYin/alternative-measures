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
    def __init__(self, c_train, t_train,l_train):
        super(MyDataset, self).__init__()
        self.c_train = c_train
        self.t_train = t_train
        self.l_train = l_train

    def __getitem__(self, index):
        return self.c_train[index], self.t_train[index],self.l_train[index]

    def __len__(self):
        return len(self.l_train)

def augment_data(seglist,labs,chorddict):
    newseglist = []
    newlablist = []
    for h in range(len(seglist)):
        newsegs = []
        newlabs = []
        #print("seglist ",seglist[h])
        for i in range(1):
            events_augmented = np.empty(0)
            
            #print('labs ',list(chorddict.values()).index(labs[h]))
            oldchord = list(chorddict.keys())[list(chorddict.values()).index(labs[h])]
            #print('oc ',oldchord)

            shiftamount = np.random.randint(1, 11)
            if oldchord == 'N' or oldchord == 'X' or oldchord == '':
                #newchord = oldchord
                continue
            else:
                oldroot,rest = oldchord.split(':')
                if rest == 'maj'  or rest == 'min':
                    newroot = (int(oldroot) + shiftamount) % 12
                    newchord =  str(newroot) + ':' + rest
                else:
                    continue
                    
            newlab = chorddict[newchord]
            #print('oldchord ',oldchord, 'newchord ',newchord,'oldlab ',labs[h],'newlab ',newlab)
            newlablist.append(newlab) 
            #print('events ',len(seglist[h])/12)
            for j in range(int(len(seglist[h][0:11])/12)):
                newbuffer = seglist[h][j*12:(j+1)*12]
                #print("nb ",newbuffer)
                newevents = np.roll(newbuffer,shiftamount)
                events_augmented = np.append(newevents,events_augmented)
            #newsegs.append(events_augmented)
            newseglist.append(events_augmented)
        #newlablist.append(newlabs)
    return(newseglist,newlablist)
    
def getfundamental(X):
    hps = copy(X)
    for h in arange(0, 3): # TODO: choose a smarter upper limit
        dec = decimate(X, h)
        hps[:len(dec)] += dec
            

def prepare_datasets(chroma_path='./data/pitch.dat',
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
    fp = open('data/chorddict.json','rb')
    chorddict = pickle.load(fp)
    print("chorddict ",chorddict)
    fp = open('data/pitch.dat','rb')
    pitches = pickle.load(fp)
    fp.close()
    fp = open('data/timbre.dat','rb')
    timbres = pickle.load(fp)
    fp.close()

    print("len all",len(pitches)," ",len(timbres))
    labs = []
    
    #print(labmatrix)
    songlist = []
    lablist = []
    labmatrix = []
    for h in range(len(pitches)):
        seglist = []
        oldchord = 'OC'
        newchord = ''
        labs = []
        for i in range(0,len(pitches[h][1])):
            newchord = pitches[h][1][i][1][2]
            if newchord == oldchord:
                eventlist = np.append(eventlist,np.asarray(pitches[h][1][i][0][2:14]))
            else:
                if i == 0:
                    pass
                else:
                    seglist.append(eventlist)
                oldchord = newchord
                eventlist = np.empty(0)
                eventlist = np.append(eventlist,np.asarray(pitches[h][1][i][0][2:14]))
                labs.append(pitches[h][1][i][1][2])
        seglist.append(eventlist)
        songlist.append(seglist)
        labmatrix.append(labs)
        newseglist,newlablist = augment_data(seglist,labs,chorddict)
    fp = open('data/bpitch.dat','rb')
    pitches1 = pickle.load(fp)
    fp.close()

    for h in range(len(pitches1)):
        seglist = []
        oldchord = 'OC'
        newchord = ''
        labs = []
        for i in range(0,len(pitches1[h][1])):
            newchord = pitches1[h][1][i][1][2]
            if newchord == oldchord:
                eventlist = np.append(eventlist,np.asarray(pitches1[h][1][i][0][2:14]))
            else:
                if i == 0:
                    pass
                else:
                    seglist.append(eventlist)
                oldchord = newchord
                eventlist = np.empty(0)
                eventlist = np.append(eventlist,np.asarray(pitches1[h][1][i][0][2:14]))
                labs.append(pitches1[h][1][i][1][2])
        seglist.append(eventlist)
        songlist.append(seglist)
        labmatrix.append(labs)
        newseglist,newlablist = augment_data(seglist,labs,chorddict)
    
    chroma_train = np.asarray(songlist)
    label_train = np.asarray(labmatrix)

    songlist = []
    lablist = []
    labmatrix = []
    for h in range(len(pitches)):
        seglist = []
        oldchord = 'OC'
        newchord = ''
        labs = []
        for i in range(0,len(pitches[h][1])):
            newchord = pitches[h][1][i][1][2]
            if newchord == oldchord:
                eventlist = np.append(eventlist,np.asarray(timbres[h][1][i][0][1:3]))
            else:
                if i == 0:
                    pass
                else:
                    seglist.append(eventlist)
                oldchord = newchord
                eventlist = np.empty(0)
                eventlist = np.append(eventlist,np.asarray(timbres[h][1][i][0][1:3]))
        seglist.append(eventlist)
        songlist.append(seglist)
    fp = open('data/btimbre.dat','rb')
    timbres = pickle.load(fp)
    fp.close()
    print("len all b",len(pitches1)," ",len(timbres))
    print(" chords ",pitches[h][1][i][1][2] )
    for h in range(len(pitches1)):
        seglist = []
        oldchord = 'OC'
        newchord = ''
        labs = []
        for i in range(0,len(pitches1[h][1])):
            newchord = pitches1[h][1][i][1][2]
            if newchord == oldchord:
                eventlist = np.append(eventlist,np.asarray(timbres[h][1][i][0][1:3]))
            else:
                if i == 0:
                    pass
                else:
                    seglist.append(eventlist)
                oldchord = newchord
                eventlist = np.empty(0)
                eventlist = np.append(eventlist,np.asarray(timbres[h][1][i][0][1:3]))
        seglist.append(eventlist)
        songlist.append(seglist)
    
    
    timbre_train = np.asarray(songlist)
    # for i in range(len(songlist)):
    #     print("sls ",len(songlist[i]))
    #     for j in range(len(songlist[i])):
    #         print("sls "," ",i," ",j," ",len(songlist[i][j]))
    
                
    
        
    
    
    
        
    X = MyDataset(chroma_train, timbre_train,label_train)

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
    #print("x ",X[train_indices][0].shape)
    # chromaobs = X[train_indices][0].reshape(-1,X[train_indices][0].shape[0])
    # timbreobs = X[train_indices][1].reshape(-1,X[train_indices][1].shape[0])
    # labs = X[train_indices][2].reshape(-1,X[train_indices][2].shape[0])

    # print("after ",chromaobs.shape)
    
    chromaobs = []
    for csample in X[train_indices][0]:
        chromaobs.append(csample)
    timbreobs = []
    for tsample in X[train_indices][1]:
        timbreobs.append(tsample)
    labs = []
    for lsample in X[train_indices][2]:
        #lsamp3 = np.concatenate([lsample,np.array([0])])
        labs.append(lsample)
    x = zip(chromaobs,timbreobs,labs)
    for l in x:
        train_set.append(list(l))
        #print("t ",len(l[0]),"c ",len(l[1]),"l ",len(l[2]))
    print("tset len XXXXXXXXXXXX ",len(train_set))
   
    
    val_set = []
    chromaobs = []
    for csample in X[val_indices][0]:
        chromaobs.append(csample)
    timbreobs = []
    for tsample in X[val_indices][1]:
        timbreobs.append(tsample)
    labs = []
    for lsample in X[val_indices][2]:
        #print("z")
        labs.append(lsample)
    #print("qqqq",len(labs))
    x = zip(chromaobs,timbreobs,labs)
    for l in x:
        val_set.append(list(l))
    #print("val labs",len(val_set[2:]))
    
    

    test_set = []
    chromaobs = []
    for csample in X[test_indices][0]:
        chromaobs.append(csample)
    timbreobs = []
    for tsample in X[test_indices][1]:
        timbreobs.append(tsample)

    labs = []
    for lsample in X[test_indices][2]:
        labs.append(lsample)
    #test_set = [timbreobs,chromaobs]
    x = zip(chromaobs,timbreobs,labs)
    for l in x:
        test_set.append(list(l))

    print("val set final ",len(val_set[1:]))
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
        
        inputs_a, inputs_b,labels = batch
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
            #print("pred ",predicted,"lba ",labels,"cor ",correct)
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
