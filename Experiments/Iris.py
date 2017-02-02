# Module Example
# Abraham Oliver, 2016
# Reverse Learning

import numpy as np
from random import randint, random
from csv import reader as CSV

def getDataset():
    # Prepare Data
    data = {"sepal length":[], "sepal width": [], "petal length": [],
            "petal width": [], "species": []}
    with open("C:/Users/abeol/Git/ReverseLearning/DataSets/iris.tab") as tsv:
        reader = CSV(tsv, dialect="excel-tab")
        for line in reader:
            if reader.line_num not in [1, 2, 3]:
                data["sepal length"].append(line[0])
                data["sepal width"].append(line[1])
                data["petal length"].append(line[2])
                data["petal width"].append(line[3])
                data["species"].append(line[4])
    # Data in training format
    xs = []
    ys = []
    for i in range(len(data["species"])):
        # Inputs
        t = [float(data["sepal length"][i]), float(data["sepal width"][i]), float(data["petal length"][i]),
             float(data["petal width"][i])]
        xs.append(t)
        # Label
        if data["species"][i] == "Iris-setosa":
            ys.append([1.0, 0.0, 0.0])
        elif data["species"][i] == "Iris-versicolor":
            ys.append([0.0, 1.0, 0.0])
        elif data["species"][i] == "Iris-virginica":
            ys.append([0.0, 0.0, 1.0])
    # Final set
    trainingData = [xs, ys]
    return trainingData

data = getDataset()

import sklearn as skl
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(data[0])
XTrain = scaler.transform(data[0])

def getClassifier(hidden = ()):
    cls = MLPClassifier(hidden_layer_sizes = hidden,
                       activation = "relu",
                       learning_rate = "constant",
                       learning_rate_init = .01,
                       solver = "lbfgs",
                       verbose = False,
                       max_iter = 10000000,
                       tol = 1e-40)
    cls.fit(XTrain, data[1])
    return cls

best = None
bestAccuracy = 0.0

def getAccuracy(classifier):
    c = classifier
    runner = 0
    for i in range(len(data[0])):
        y_ = c.predict([XTrain[i]])[0]
        compl = 0
        for j in range(len(y_)):
            if y_[j] == data[1][i][j]: compl += 1
        if compl == len(y_): runner += 1
    return float(runner) / len(data[0])

# for qq in range(6, 8):
qq = 20
c = getClassifier((qq))
# Find accuracy
acc = getAccuracy(c)
print("Accuracy w/ H{0} :: {1}".format(qq, acc))
if acc > bestAccuracy:
    bestAccuracy = acc
    best = c
    print("BEST ^^^^^^")

print(best.coefs_)
print(best.intercepts_)
print(best.loss_)


"""
Accuracy w/ H20 :: 1.0
[array([[ -1.55208356e+00,   1.18757620e+00,   4.12500626e-01,
          1.16681960e-01,   1.55607526e-01,  -5.45480132e-01,
         -4.75995937e-01,   5.46375977e-01,   1.15451046e-01,
         -1.09867723e-01,   3.16660899e-01,  -1.00577768e+00,
          1.25233636e+00,   3.24015994e-01,  -1.20380885e+00,
          6.30533626e-01,   2.14889671e+00,  -4.55190338e-01,
          3.74832018e+00,  -2.35737263e-01],
       [  1.87760555e-01,  -2.87996083e+00,   5.58014486e-01,
          2.61049963e-01,   7.55605920e-01,   1.29896134e-01,
         -1.32748644e+00,  -1.78488614e+00,  -1.96792644e-02,
         -1.20463766e-01,  -1.73702261e+00,  -1.63478129e-02,
         -3.32296412e+00,   2.19423288e-01,   3.09379823e-01,
         -4.04871809e-01,  -7.77252709e-01,   1.41492774e-02,
         -5.28766834e+00,   1.29901345e-01],
       [ -2.41116958e+00,   1.70154507e+00,   1.01442306e+00,
         -2.75218609e-01,   3.30497342e-01,  -1.12832281e+00,
         -1.43903117e-02,   1.22983369e+00,  -1.51071332e-01,
         -3.34519044e-02,   7.55643793e-01,  -1.57172764e+00,
          2.03218577e+00,   1.37740458e+00,  -2.12426874e+00,
          7.23908810e-01,   1.57665058e+01,  -5.32366061e-01,
          1.35567540e+00,  -5.92669984e-02],
       [ -2.17893296e+00,   4.31650752e+00,  -6.08514489e-01,
         -2.57248629e-01,   1.56030200e-01,  -9.86858060e-01,
          1.26756043e+00,   2.20065776e+00,   1.78769344e-02,
         -1.56240566e-01,   1.90655165e+00,  -1.39545457e+00,
          5.17203104e+00,  -4.72981217e-01,  -1.88484396e+00,
          9.92328964e-01,  -1.17042273e+01,  -5.17855278e-01,
         -5.53623135e+00,  -7.99879446e-02]]), array([[  1.56545501,  -3.40803675,  -0.76285757],
       [ -0.68563316,  -3.73505475,   3.53843373],
       [ -0.69192319,   1.00945949,  -0.8288276 ],
       [  0.18753637,  -0.05057133,  -0.11768974],
       [ -0.46499695,  -0.51232483,   0.45032704],
       [  0.74210304,  -1.51341826,  -0.15682969],
       [ -1.12031036,  -1.23562776,   0.91028382],
       [ -0.23500085,  -2.03625619,   1.98499866],
       [  0.0807408 ,   0.07640395,   0.04954814],
       [ -0.14495245,  -0.20513206,  -0.01910432],
       [ -0.31843199,  -1.55383307,   1.62396709],
       [  0.87546148,  -2.11757044,  -0.36464311],
       [ -0.18578165,  -4.21958215,   4.38561333],
       [ -1.06625204,   1.11718164,  -0.66903235],
       [  1.49180455,  -2.78267689,  -0.27651205],
       [ -0.48746378,  -1.24231183,   0.64328064],
       [  0.20416907, -13.90016119,  13.39773124],
       [  0.35071435,  -0.81982917,  -0.11052024],
       [ -1.24518378,   5.5775705 ,  -5.44018137],
       [ -0.14077251,  -0.1145499 ,   0.03255091]])]
[array([ 0.72757985, -3.42786844,  5.93314958, -1.474309  , -2.57056925,
        0.58258358,  1.91282172, -1.53389082, -0.51565568, -0.1956557 ,
       -1.0551703 ,  0.31256642, -3.66495477,  4.91928058,  0.95494726,
       -0.97317538, -6.89476605,  0.10638528,  2.12729261, -1.34720481]), array([-2.94639577,  3.22891037, -5.87752704])]"""