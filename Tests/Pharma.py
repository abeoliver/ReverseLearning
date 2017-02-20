# Pharma Dataset Tests
# Reverse Learning, 2017

from random import random, randint, uniform
import pandas

def getPoints(amount, error):
    def f(x): return -0.0817 * (x - 60) *  (x + 10)
    xs = []
    ys = []
    for i in range(amount):
        newX = uniform(3, 58)
        effective = f(newX)
        newY = uniform(effective - error, effective + error)
        xs.append(newX)
        ys.append(round(newY))
    d = {"effectiveness": ys, "age": xs}
    df = pandas.DataFrame(d)
    df.to_csv("test", sep="\t")

getPoints(1000, 20)