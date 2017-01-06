# Algorithm Playground
# ReverseLearning

from Network import Network

# Import Dataset
import csv
with open("../DataSets/OnlineNewsPopularity/OnlineNewsPopularity.csv", 'rb') as f:
    reader = csv.reader(f)
    for row in reader:
        print row