import sys
import numpy
import pandas
import tensorflow as tf

## Input: a csv file of 
YX = pandas.read_csv( sys.argv[1] )     ## read in data
Y = pandas.get_dummies(YX[YX.columns[0]])   ## transform data
