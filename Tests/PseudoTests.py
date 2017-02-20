# Pharma Dataset Tests
# Reverse Learning, 2017

from random import random, randint, uniform
import pandas

def getPoints(function, amount, error, filename):
    xs = []
    ys = []
    for i in range(amount):
        newX = uniform(3, 58)
        effective = function(newX)
        newY = uniform(effective - error, effective + error)
        xs.append(newX)
        ys.append(round(newY))
    d = {"x": xs, "y": ys}
    df = pandas.DataFrame(d)
    df.to_csv("test", sep="\t")