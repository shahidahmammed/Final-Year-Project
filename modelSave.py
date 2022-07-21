# Miss. Shahidah Kunhi Endiape Mammed
# BSc. (Hons) in Computer Science
# TP055203
# UC3F2111CS
# Description: Trains and saves the modules
# Starting date: 06/06/2022
# Modified date: 16/07/2022

# import all libraries
from featureExtraction import dataSet
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import  GradientBoostingClassifier, RandomForestClassifier
from sklearn.svm import SVC
import pickle
import os

# train the dataset
trainingFeatures, testingFeatures, trainingEmotions, testingEmotions = dataSet(testSize=0.25, noise=True) 

# print number of trained, tested and features extracted 
print("The number of samples trained: ", trainingFeatures.shape[0])
print("The number of samples tested: ", testingFeatures.shape[0])
print("The number of features extracted: ", trainingFeatures.shape[1])

# store all the classifier (machine learning models) in a list
classifiers = [ 
    MLPClassifier(alpha=0.01, batch_size=256, epsilon=1e-08, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500),
    SVC(C=0.001, gamma=0.001, kernel='poly'),
    KNeighborsClassifier(weights='distance', n_neighbors=6, p=1),
    RandomForestClassifier(max_depth=7, max_features=0.5, min_samples_leaf=1, min_samples_split=2, n_estimators=40),
    GradientBoostingClassifier(learning_rate=0.3, max_depth=7, max_features=None, min_samples_leaf=1, min_samples_split=2, n_estimators=70)
]

# AllModels store all the models with its respective accuracy scores
AllModels = [] 

# loop through the classifiers
for classifier in classifiers: 
    print("The ", classifier.__class__.__name__, " model is being trained") # print out the name of classifier
    
    model = classifier.fit(trainingFeatures, trainingEmotions) # train the model 
    accTrain = model.score(trainingFeatures, trainingEmotions) # get the training accuracy score
    accTest = model.score(testingFeatures, testingEmotions) # get the accuracy accuracy score

    print("training set score: {:.2f}".format(accTrain*100)) # print the training accuracy score
    print("testing set score: {:.2f}".format(accTest*100)) # print the training accuracy score

    if not os.path.isdir("model/{}".format(classifier.__class__.__name__)): # check if not path exists
        os.mkdir("model/{}".format(classifier.__class__.__name__)) # make path if no path exists
    pickle.dump(model, open("model/{}/6EMO6{}_classifier.model".format(model.__class__.__name__, model.__class__.__name__), "wb")) # save each model
    
    AllModels.append((model, accTrain, accTest)) # add each model to AllModels list
    
pickle.dump(AllModels, open("model/6EMO6_models.pickle", "wb")) # save all the models