from src.algorithms import NaiveBayes, Svm

naive_bayes = NaiveBayes()
svm = Svm()
print(naive_bayes.test(), svm.test())
