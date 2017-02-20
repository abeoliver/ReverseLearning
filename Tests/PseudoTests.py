# Pharma Dataset Tests
# Reverse Learning, 2017

from random import random, randint, uniform
import pandas

def getPoints(function, amount, error, filename, roundX = False, roundY = False):
    xs = []
    ys = []
    for i in range(amount):
        newX = uniform(3, 58)
        if roundX: newX = round(newX)
        effective = function(newX)
        newY = uniform(effective - error, effective + error)
        if roundY: newY = round(newY)
        xs.append(newX)
        ys.append(newY)
    d = {"x": xs, "y": ys}
    df = pandas.DataFrame(d)
    df.to_csv("test", sep="\t")