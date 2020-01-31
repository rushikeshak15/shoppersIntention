# ShoppersIntention

Objective – To predict whether or not a visitor to the website would generate revenue
for it. Use the Online Shopper’s Purchasing Intention Dataset.

Explaination –

a)Pre-processing -
Data preprocessing is a data mining technique that involves transforming raw
data into an understandable format.

1.Null Values Removal -
If the missing values are not handled properly by the developer, then he/she may end
up drawing an inaccurate inference about the data. In our code, we have replaced these
values with 0.

2.One-hot encoding -
A categorical value represents the numerical value of the entry in the dataset. For
example, in our dataset, there are 8 different values for OperatingSystems. Each of
these numbers represents a different OS. However, our algorithm might perceive a
value 8 to be greater than 3. In order to fix this, we require one-hot encoding to obtain
dummy variables. A new column is created for each category. In our example, the
feature OS is removed and new features are added. These are OperatingSystems_1,
OperatingSystems_2, ... , OperatingSystems_8.

3.Reduction of highly correlated features -
If multiple features are highly correlated, we remove some of them. This helps reduces
harmful bias, and makes training the algorithm faster.

4.Feature Scaling -
It basically helps to normalize the data within a particular range. The goal of
normalization is to change the values of numeric columns in the dataset to a common
scale, without distorting differences in the ranges of values. This makes gradients
converge quicker. In our project, we have used min-max scaling.
We have used the default max(x) and min(x) values of sklearn’s MinMax(), 1 and 0.

b)Cross-validation –

Cross-validation is a statistical method used to estimate the skill of machine learning
models. To evaluate the performance of any machine learning model we need to test it
on some unseen data. Based on the models performance on unseen data we can say
weather our model is Under-fitting/Over-fitting/Well generalised.

1.K-folds cross-validation –
The procedure has a single parameter called k that refers to the number of groups that
a given data sample is to be split into. It results in a less biased model compare to other
methods. Because it ensures that every observation from the original dataset has the
chance of appearing in training and test set.

#Prediction Algorithms -
Logistic Regression -
Logistic regression is named for the function used at the core of the method, the logistic
function, also called the sigmoid function.
It’s an S-shaped curve that can take any real-valued number and map it into a value
between 0 and 1, but never exactly at those limits. The predictions are calculated as
follows:
y = e^(b0 + b1*x) / (1 + e^(b0 + b1*x))

Gaussian Naive Bayes -
Bayes theorem works on conditional probability . Conditional probability is the
probability that something will happen, given that something else has already occurred .
Naive Bayes is a kind of classifier which uses the Bayes Theorem. It predicts
membership probabilities for each class such as the probability that given record or
data point belongs to a particular class. The class with the highest probability is
considered as the most likely class.

Gradient Boosting Classifier -
Gradient boosting classifiers are a group of machine learning algorithms that combine
many weak learning models together to create a strong predictive model. Decision trees
are usually used when doing gradient boosting. The idea behind "gradient boosting" is
to take a weak hypothesis or weak learning algorithm and make a series of tweaks to it
that will improve the strength of the hypothesis/learner. Gradient boosting systems use
decision trees as their weak learners. Regression trees are used for the weak learners,
and these regression trees output real values. Because the outputs are real values, as
new learners are added into the model the output of the regression trees can be added
together to correct for errors in the predictions.

SVC -
In this algorithm, we plot each data item as a point in n-dimensional space (where n is
number of features you have) with the value of each feature being the value of the
particular coordinate. Then, we need to perform classification by finding the hyper-
plane that differentiate the two classes very well. Support Vectors are simply the co-
ordinates of individual observation. Support Vector Machine is a frontier which best
segregates the two classes (hyper-plane/ line).

Random Forest -
Random forest, like its name implies, consists of a large number of individual decision
trees that operate as an ensemble. Each individual tree in the random forest spits out a
class prediction and the class with the most votes becomes our model’s prediction.
