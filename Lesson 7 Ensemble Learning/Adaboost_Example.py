from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

#model = AdaBoostClassifier() # default

model = AdaBoostClassifier(base_estimator = DecisionTreeClassifier(max_depth=2), n_estimators = 4)
# base_estimator: The model utilized for the weak learners 
#    (Warning: Don't forget to import the model that you decide to use for the weak learner).
# n_estimators: The maximum number of weak learners used.
#model.fit(x_train, y_train)
#model.predict(x_test)

