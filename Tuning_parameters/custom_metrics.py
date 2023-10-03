
"""
Functions for custom classification metrics 
"""

# Import libraries 

from sklearn.metrics import confusion_matrix

# Accuracy per class 
def astro_acc(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    acc_c = cm.diagonal()
    return acc_c[0] #astrocytes acc

def neuron_acc(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    acc_c = cm.diagonal()
    return acc_c[1] #neuron acc

def oligo_acc(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    acc_c = cm.diagonal()
    return acc_c[2] #oligo acc

def others_acc(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    acc_c = cm.diagonal()
    return acc_c[3] #ignore acc



## Confusion per class: 

## Astro
def A_as_N(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[0][1] # percentage that A is wrongly classified as N   

def A_as_O(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[0][2] # percentage that A is wrongly classified as O       


def A_as_Others(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[0][3]   

##Neurons 

def N_as_A(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[1][0] 

def N_as_O(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[1][2] 

def N_as_Others(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[1][3] 


## Oligo 

def O_as_A(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[2][0] 

def O_as_N(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[2][1] 

def O_as_Others(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[2][3] 

## Others 

def Others_as_A(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[3][0] 

def Others_as_N(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[3][1] 

def Others_as_O(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"],normalize='true')
    return cm[3][2] 

## Functions for custom classification metrics: RAW VALUES 


## Confusion per class: 

## Astro
def A_as_N_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[0][1] # percentage that A is wrongly classified as N   

def A_as_O_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[0][2] # percentage that A is wrongly classified as O       


def A_as_Others_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[0][3]   

##Neurons 

def N_as_A_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[1][0] 

def N_as_O_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[1][2] 

def N_as_Others_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[1][3] 


## Oligo 

def O_as_A_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[2][0] 

def O_as_N_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[2][1] 

def O_as_Others_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[2][3] 

## Others 

def Others_as_A_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[3][0] 

def Others_as_N_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[3][1] 

def Others_as_O_r(clf,X,y): 
    y_pred = clf.predict(X)
    cm = confusion_matrix(y,y_pred,labels=["Astro","Neuron","Oligo","Others"])
    return cm[3][2] 

