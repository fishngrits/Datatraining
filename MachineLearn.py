#import data
from sklearn import datasets

iris = datasets.load_iris()

X = iris.data
Y = iris.target

#spiltting data into two one for training and one for testing
from sklearn.model_selection import  train_test_split
#train is for the training data and test is for test data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.5)

#gooing to classifier decision tree
from sklearn import  tree

my_decision_tree_classifier = tree.DecisionTreeClassifier()

#now it will train classifier off of the training data
my_decision_tree_classifier.fit(X_train, Y_train)
#predict method to classif data
predictions_from_decision_tree_classifier = my_decision_tree_classifier.predict(X_test)
#printing out the prediction