import pandas as pd
import pickle
#import _pickle as cPickle
import cPickle
from numpy import genfromtxt
import numpy as np
from sklearn import metrics


from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif

#mydataF = pd.read_csv('Features_All.csv')
#mydataL = pd.read_csv('Target_All.csv')

mydata = pd.read_csv('snowall.csv')

df1 = pd.DataFrame(mydata, columns=['duration','airtemp','windspeed','visibility','snowfall','snowrate','districtarea'])
df2 = pd.DataFrame(mydata,columns=['Average'])

from sklearn import preprocessing
#features = df1.apply(preprocessing.LabelEncoder().fit_transform)
#labels = df2.apply(preprocessing.LabelEncoder().fit_transform)
#features = features.values
#labels = labels.values
features = df1.values
labels = df2.values
labels =labels.ravel()
print (type(features))
print (features.shape)
print (features[0:3,:])
print (labels.shape)
print (labels[0:3])
#print (labels)
### test_size is the percentage of events assigned to the test set
### (remainder go into training)
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.1, random_state=42)

##SVM
from sklearn import svm
from sklearn.metrics import accuracy_score
clasf = svm.SVC(C=100,kernel = 'linear')


from sklearn.metrics import confusion_matrix
#print (confusion_matrix(pred, labels_test))


from sklearn.model_selection import cross_val_score
scores = cross_val_score(clasf, features, labels[:], cv=10)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(clasf, features, labels[:], cv=10)
print (metrics.accuracy_score(labels[:], predicted))
print (confusion_matrix(labels[:], predicted))

from sklearn.metrics import precision_score
print "Precision", precision_score(labels[:], predicted, average=None)
print "Precision-macro", precision_score(labels[:], predicted, average='macro')
print "Precision-micro", precision_score(labels[:], predicted, average='micro')
from sklearn.metrics import recall_score
print "Recall", recall_score(labels[:], predicted, average=None)
print "Recall-macro", recall_score(labels[:], predicted, average='macro')
print "Recall-micro", recall_score(labels[:], predicted, average='micro')
from sklearn.metrics import f1_score
print "F1-score", f1_score(labels[:], predicted, average=None)
print "F1-score-macro", f1_score(labels[:], predicted, average='macro')
print "F1-score-micro", f1_score(labels[:], predicted, average='micro')
##plot
#import pydotplus
#import graphviz
#from IPython.display import Image


#dot_data = tree.export_graphviz(clf, out_file=None)
#graph = pydotplus.graph_from_dot_data(dot_data)
#graph.write_pdf("tree.pdf") 
#Image(graph)




