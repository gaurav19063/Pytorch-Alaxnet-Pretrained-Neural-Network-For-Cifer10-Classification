from sklearn.manifold import TSNE
import h5py
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pickle
from sklearn import metrics
import time as time
from sklearn.neural_network import MLPClassifier
from struct import unpack

def one_hot_encode(d):
    d = np.array(pd.get_dummies(d))
    d = d.transpose()
    return d

def ExtractData():

    data=h5py.File('MNIST_Subset.h5', 'r')

    a = np.array(data['X'][:])
    y = np.array(data['Y'][:])
    x = a.swapaxes(1, 2).reshape(14251, -1) / 255
    return train_test_split(x, y, test_size=0.25, stratify=y, random_state=42)



def initw(a, m):
    return  np.random.randn(a, m)
def initb(a, l):
    return np.random.randn(a, l)




def parametersInit(x_train,arr):
    n=len(arr)
    w=[]
    b=[]
    m=len(x_train[0])

    for i in range (4):
        a = arr[i]
        w.append(initw(a, m))
        b.append(initb(a, x_train.shape[0]))

        m = a

    return w, b
def makelist(t):
    p=list()
    p=p.append(t.transpose())
    return p
def softmax(x):
    a= np.exp(x)
    b=np.sum(np.exp(x), axis=0)
    return a/b
def cost_m(y_train,l):
    y1 = np.multiply(y_train, np.log(l[4]))
    y2 = np.multiply(1 - y_train, np.log(1 - l[4]))
    return (np.sum(y1 + y2) / (-x_train.shape[0]))

def forward_propagation(x_train, y_train, w, b):

    l =list()
    l.append(x_train.transpose())

    for i in range(3):
        p = np.dot(w[i],l[i])
        q = p + b[i]
        l.append(1 / (1 + np.exp(-q)))

    p = np.dot(w[-1], l[3])
    q = p + b[-1]
    l.append(softmax(q))
    return l, cost_m(y_train,l)



def trainWithBackProp(a,y, w, b, cost):
    da = []
    dz = []
    print('Error :', cost)
    da.append((a[-1] - y).transpose())
    for i in range(len(a)-1):
        dz.append(np.dot(da[i], w[-(i+1)]))
        da.append(dz[i]*((a[-(i+2)])*(1-(a[-(i+2)])).transpose()))
        # print(da[i].shape, a[i].shape, i)
    da.reverse()
    dz.reverse()

    for i in range(len(w)):
        # print( a[i].shape,da[i+1].shape, w[i].shape)
        w[i] -= 0.00001 * np.dot(a[i], da[i+1]).transpose()
        b[i] -= 0.00001 * (np.sum(da[i+1].transpose(), axis=1, keepdims=True))/da[i+1].shape[0]

    # print(w, b)
    return w, b





def weigthsplot(weights):

    for w in weights:
        start = time.time()
        new_w = TSNE(n_components=2).fit_transform(w)
        print(new_w.shape)
        print("time:-", time.time() - start)
        plots(new_w)


def plots(x):
    plt.scatter(x[:, 0], x[:, 1])
    plt.show()

















# ------------------------------------------------------------
x_train,x_val,y_train,y_val=ExtractData()
x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, stratify= y_train)
arr = [100, 50, 50, 2]
w,b=parametersInit(x_train,arr)
output, cost = forward_propagation(x_train, y_train, w, b)
print (output,cost)
start = time.time()
y_train = one_hot_encode(y_train)
y_val = one_hot_encode(y_val)
y_train=y_train.transpose()
print(x_train.shape,y_train.shape)
print(output.shape,cost.shape)

l = []
acc = []
vl = []
va = []
for i in range(1000):
    print(i)
    out, cost = forward_propagation(x_train, y_train, w, b)
    w, b = trainWithBackProp(out, y_train, w, b, cost)
    l.append(cost)
    y_pred = [0 if out[-1][0][i] <= 0.5 else 1 for i in range(len(a[-1][0]))]
    print("acc", metrics.accuracy_score(y_train[0], y_pred))
    acc.append(metrics.accuracy_score(y_train[0], y_pred))

    out, cost = forward_propagation(x_val, y_val, w, b)
    vl.append(cost)
    y_pred = [0 if a[-1][0][i] <= 0.5 else 1 for i in range(len(a[-1][0]))]
    print("Val acc", metrics.accuracy_score(y_val[0], y_pred))
    va.append(metrics.accuracy_score(y_val[0], y_pred))

plt.plot([v for v in range(1000)], acc)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Train Accuracy vs iterations")
plt.show()

plt.plot([v for v in range(1000)], l)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Train loss vs iterations")
plt.show()

plt.plot([v for v in range(1000)], va)
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy vs iterations")
plt.show()

plt.plot([v for v in range(1000)], vl)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Validation loss vs iterations")
plt.show()

for wi in w:
    start = time.time()
    new_w = TSNE(n_components=2).fit_transform(wi)
    print(new_w.shape)
    print("time:-", time.time() - start)
    plt.scatter(new_w[:, 0], new_w[:, 1])
    plt.show()


a, cost = forward_propagation(x_test, y_test, w, b)
print("cost = ", cost)
y_pred = [0 if a[-1][0][i] < 0.5 else 1 for i in range(len(a[-1][0]))]
print("test=", metrics.accuracy_score(y[0], y_pred))


