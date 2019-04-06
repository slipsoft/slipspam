from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import csv

# creating labelEncoder
le = preprocessing.LabelEncoder()
labels = []
features = []
with open('./Data/spambase.data') as csvDataFile:
    csvReader = csv.reader(csvDataFile)
    for row in csvReader:
        labels.append(int(row.pop()))
        features.append(list(map(lambda x: float(x), row)))

# print(features[:20])
# print(labels[:20])

# Create a Gaussian Classifier
model = GaussianNB()

# Train the model using the training sets
model.fit(features, labels)

# Predict Output
predicted = model.predict([[0.0, 0.42, 0.42, 0.0, 1.27, 0.0, 0.42, 0.0, 0.0, 1.27, 0.0, 0.0, 0.0, 0.0, 0.0, 1.27, 0.0, 0.0, 1.7, 0.42, 1.27, 0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.27, 0.0, 0.0, 0.42, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.063, 0.0, 0.572, 0.063, 0.0, 5.659, 55.0, 249.0]])
print("Predicted Value:", predicted)
