import pandas as pd import numpy as np import matplotlib.pyplot as plt %matplotlib inline
#Algo SVM
fichier = open( "spambase.data", "r")
txt = fichier.read()

while   

fichier.close()
b = []
line = txt.split("\n")[0] # 1ere ligne de donnees
line = line.split(";") # le separateur utilise a l'ecriture
b.append(float(line[0]))
b.append(int(line[1]))
print "b:", b