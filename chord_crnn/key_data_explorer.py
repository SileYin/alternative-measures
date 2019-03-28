import plotly
import numpy as np
import plotly.graph_objs as go
import operator
import scipy.stats
from sklearn import svm

value_path='data/chroma_train_key.npy'
label_path='data/label_train_key.npy'
vals = np.load(value_path)
labels = np.load(label_path)
print(vals.shape)
X = vals[0:9000]
print("x ",X)
roots = []
#roots = np.zeros((X.shape[0],1000),dtype=int)
for song in range(len(X)):
    songroots = []
    for row in range(len(X[song])):
        root = []
        root = np.argmax(vals[song][row])
        songroots.append(root)
    #print("xxxxxxx ",len(songroots))
    roots.append(songroots)
print("roots ",roots[0],"len ",len(roots))
most_common = np.empty(len(roots))
print("mc ",most_common.shape)
for song in range(len(roots)):
    #print("xxx ",song," ",roots[song])
    foo = np.array(roots[song]).reshape(-1)
    #print("shpa ",foo.shape)
    mc = np.bincount(foo).argmax()
    most_common[song] = mc

y = labels[0:9000]
testX = vals[9001:]
testy = labels[9001:]
#testroots = np.zeros((testX.shape[0],1000),dtype=int)
testroots = []
 
for song in range(len(testX)):
    songroots = []
    for row in range(len(testX[song])):
        root = []
        root = np.argmax(testX[song][row])
        songroots.append(root)
    testroots.append(songroots)
test_most_common = np.empty(len(testroots))
for song in range(len(testroots)):
    test_most_common[song] = np.bincount(testroots[song]).argmax()
print("mc ",most_common[0:11])
print("y  ",y.reshape(-1)[0:11])
print("X ",most_common.shape)
print("l ",y.shape,)

#for dims in range(roots.shape[1]):
popsl,popint,popr,popp,popstd = scipy.stats.linregress(most_common.reshape(-1),y.reshape(-1))
print("sl ",popsl,"int ",popint,"r ",popr,"stderr ",popstd)
lin_clf = svm.LinearSVC(max_iter=10000)
lin_clf.fit(most_common.reshape(len(X),1), y.reshape(-1)) 
# LinearSVC(C=1.0, class_weight=None, dual=True, fit_intercept=True,
#      intercept_scaling=1, loss='squared_hinge', max_iter=1000,
#      multi_class='ovr', penalty='l2', random_state=None, tol=0.0001,
#      verbose=0)
dec = lin_clf.decision_function(test_most_common.reshape(len(testX),1))
print(dec)
pred = lin_clf.predict(test_most_common.reshape(len(testX),1))
print("pred ",pred[0:20])
print("y    ",testy.reshape(-1)[0:20])
print("pred.shape ",pred.shape,"y shape ",testy.shape)
correct = (pred.reshape(-1) == testy.reshape(-1)).sum().item()

print("??? ",correct/5000)
