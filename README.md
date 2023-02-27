# CREDIT-CARD-PROJECT

# Project Title

CREDIT CARD FILE

DESCRIPTION:

In our credit card fraud detection project, we’ll use Python, one of the most popular programming languages available. Our solution would detect if someone bypasses the security walls of our system and makes an illegitimate transaction.  

The dataset has credit card transactions, and its features are the result of PCA analysis. It has ‘Amount’, ‘Time’, and ‘Class’ features where ‘Amount’ shows the monetary value of every transaction, ‘Time’ shows the seconds elapsed between the first and the respective transaction, and ‘Class’ shows whether a  transaction is legit or not. 





Step 1: Import Packages
We’ll start our credit card fraud detection project by installing the required packages. Create a ‘main.py’ file and import these packages:

import numpy as np

import pandas as pd

import sklearn

from scipy.stats import norm

from scipy.stats import multivariate_normal

from sklearn.preprocessing import MinMaxScaler

import matplotlib.pyplot as plt

import seaborn as sns

Step 2: Look for Errors
Before we use the dataset, we should look for any errors and missing values in it. The presence of missing values can cause your model to give faulty results, rendering it inefficient and ineffective. Hence, we’ll read the dataset and look for any missing values:

df = pd.read_csv(‘creditcardfraud/creditcard.csv’)

# missing values

print(“missing values:”, df.isnull().values.any())

We found no missing values in this dataset, so we can proceed to the next step. 

Join the Artificial Intelligence Course online from the World’s top Universities – Masters, Executive Post Graduate Programs, and Advanced Certificate Program in ML & AI to fast-track your career.

Step 3: Visualization
In this step of our credit card fraud detection project, we’ll visualize our data. Visualization helps in understanding what our data shows and reveals any patterns which we might have missed. Let’s create a plot of our dataset: 

# plot normal and fraud

count_classes = pd.value_counts(df[‘Class’], sort=True)

count_classes.plot(kind=’bar’, rot=0)

plt.title(“Distributed Transactions”)

plt.xticks(range(2), [‘Normal’, ‘Fraud’])

plt.xlabel(“Class”)

plt.ylabel(“Frequency”)

plt.show()

In our plot, we found that the data is highly imbalanced. This means we can’t use supervised learning algorithms as it will result in overfitting. Furthermore, we haven’t figured out what would be the best method to solve our problem, so we’ll perform more visualisation. Use the following to plot the heatmap: 

# heatmap

sns.heatmap(df.corr(), vmin=-1)

plt.show()

Now, we’ll create data distribution graphs to help us understand where our data came from: 

fig, axs = plt.subplots(6, 5, squeeze=False)

for i, ax in enumerate(axs.flatten()):

   ax.set_facecolor(‘xkcd:charcoal’)

   ax.set_title(df.columns[i])

   sns.distplot(df.iloc[:, i], ax=ax, fit=norm,

                color=”#DC143C”, fit_kws={“color”: “#4e8ef5”})

   ax.set_xlabel(”)

fig.tight_layout(h_pad=-1.5, w_pad=-1.5)

plt.show()

With data distribution graphs, we found that nearly every feature comes from Gaussian distribution except ‘Time’. 

So we’ll use multivariate Gaussian distribution to detect fraud. As only the ‘Time’ feature comes from the bimodal distribution (and note gaussian distribution), we’ll discard it. Moreover, our visualisation revealed that the ‘Time’ feature doesn’t have any extreme values like the others, which is another reason why we’ll discard it. 

Add the following code to drop the features we discussed and scale others: 

classes = df[‘Class’]

df.drop([‘Time’, ‘Class’, ‘Amount’], axis=1, inplace=True)

cols = df.columns.difference([‘Class’])

MMscaller = MinMaxScaler()

df = MMscaller.fit_transform(df)

df = pd.DataFrame(data=df, columns=cols)

df = pd.concat([df, classes], axis=1)

Step 4: Splitting the Dataset
Create a ‘functions.py’ file. Here, we’ll add functions to implement the different stages of our algorithm. However, before we add those functions, let’s split our dataset into two sets, the validation set and the test set. 

import pandas as pd

import numpy as np

def train_validation_splits(df):

   # Fraud Transactions

   fraud = df[df[‘Class’] == 1]

   # Normal Transactions

   normal = df[df[‘Class’] == 0]

   print(‘normal:’, normal.shape[0])

   print(‘fraud:’, fraud.shape[0])

   normal_test_start = int(normal.shape[0] * .2)

   fraud_test_start = int(fraud.shape[0] * .5)

   normal_train_start = normal_test_start * 2

   val_normal = normal[:normal_test_start]

   val_fraud = fraud[:fraud_test_start]

   validation_set = pd.concat([val_normal, val_fraud], axis=0)

   test_normal = normal[normal_test_start:normal_train_start]

   test_fraud = fraud[fraud_test_start:fraud.shape[0]]

   test_set = pd.concat([test_normal, test_fraud], axis=0)

   Xval = validation_set.iloc[:, :-1]

   Yval = validation_set.iloc[:, -1]

   Xtest = test_set.iloc[:, :-1]

   Ytest = test_set.iloc[:, -1]

   train_set = normal[normal_train_start:normal.shape[0]]

   Xtrain = train_set.iloc[:, :-1]

   return Xtrain.to_numpy(), Xtest.to_numpy(), Xval.to_numpy(), Ytest.to_numpy(), Yval.to_numpy()

Step 5: Calculate Mean and Covariance Matrix
The following function will helps us calculate the mean and the covariance matrix:

def estimate_gaussian_params(X):

   “””

   Calculates the mean and the covariance for each feature.

   Arguments:

   X: dataset

   “””

   mu = np.mean(X, axis=0)

   sigma = np.cov(X.T)

   return mu, sigma

FYI: Free nlp course!

Step 6: Add the Final Touches
In our ‘main.py’ file, we’ll import and call the functions we implemented in the previous step for every set:

(Xtrain, Xtest, Xval, Ytest, Yval) = train_validation_splits(df)

(mu, sigma) = estimate_gaussian_params(Xtrain)

# calculate gaussian pdf

p = multivariate_normal.pdf(Xtrain, mu, sigma)

pval = multivariate_normal.pdf(Xval, mu, sigma)

ptest = multivariate_normal.pdf(Xtest, mu, sigma)

Now we have to refer to the epsilon (or the threshold). Usually, it’s best to initialise the threshold with the pdf’s minimum value and increase with every step until you reach the maximum pdf while saving every epsilon value in a vector.

After we create our required vector, we make a ‘for’ loop and iterate over the same. We compare the threshold with the pdf’s values that generate our predictions in every iteration. 

We also calculate the F1 score according to our ground truth values and the predictions. If the found F1 score is higher than the previous one, we override a ‘best threshold’ variable. 

Keep in mind that we can’t use ‘accuracy’ as a metric in our credit card fraud detection project. That’s because it would reflect all the transactions as normal with 99% accuracy, rendering our algorithm useless. 

We’ll implement all of the processes we discussed above in our ‘functions.py’ file:

def metrics(y, predictions):

   fp = np.sum(np.all([predictions == 1, y == 0], axis=0))

   tp = np.sum(np.all([predictions == 1, y == 1], axis=0))

   fn = np.sum(np.all([predictions == 0, y == 1], axis=0))

   precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0

   recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0

   F1 = (2 * precision * recall) / (precision +

                                    recall) if (precision + recall) > 0 else 0

   return precision, recall, F1

def selectThreshold(yval, pval):

   e_values = pval

   bestF1 = 0

   bestEpsilon = 0

   for epsilon in e_values:

       predictions = pval < epsilon

       (precision, recall, F1) = metrics(yval, predictions)

       if F1 > bestF1:

           bestF1 = F1

           bestEpsilon = epsilon

   return bestEpsilon, bestF1

In the end, we’ll import the functions in the ‘main.py’ file and call them to return the F1 score and the threshold. It will allow us to evaluate our model on the test set:

(epsilon, F1) = selectThreshold(Yval, pval)

print(“Best epsilon found:”, epsilon)

print(“Best F1 on cross validation set:”, F1)

(test_precision, test_recall, test_F1) = metrics(Ytest, ptest < epsilon)

print(“Outliers found:”, np.sum(ptest < epsilon))

print(“Test set Precision:”, test_precision)

print(“Test set Recall:”, test_recall)

print(“Test set F1 score:”, test_F1)

Here are the results of all this effort: 

Best epsilon found: 5e-324

Best F1 on cross validation set: 0.7852998065764023

Outliers found: 210

Test set Precision: 0.9095238095238095

Test set Recall: 0.7764227642276422


Test set F1 score: 0.837719298245614


Popular AI and ML Blogs & Free Courses
IoT: History, Present & Future	Machine Learning Tutorial: Learn ML	What is Algorithm? Simple & Easy
Robotics Engineer Salary in India : All Roles	A Day in the Life of a Machine Learning Engineer: What do they do?	What is IoT (Internet of Things)
Permutation vs Combination: Difference between Permutation and Combination	Top 7 Trends in Artificial Intelligence & Machine Learning	Machine Learning with R: Everything You Need to Know
AI & ML Free Courses
Introduction to NLP	Fundamentals of Deep Learning of Neural Networks	Linear Regression: Step by Step Guide
Artificial Intelligence in the Real World	Introduction to Tableau	Case Study using Python, SQL and Tableau
Conclusion
There you have it – a fully functional credit card fraud detection project!

If you have any questions or suggestions regarding this project, let us know by dropping a comment below. We’d love to hear from you. 

With all the learnt skills you can get active on other competitive platforms as well to test your skills and get even more hands-on. If you are interested to learn more about the course, check out the page of the Execitive PG Program in Machine Learning & AI and talk to our career counsellor for more information.

What is the aim of the credit card fraud detection project?
The aim of this project is to predict whether a credit card transaction is fraudulent or not, based on the transaction amount, location and other transaction related data. It aims to track down credit card transaction data, which is done by detecting anomalies in the transaction data. Credit card fraud detection is typically implemented using an algorithm that detects any anomalies in the transaction data and notifies the cardholder (as a precautionary measure) and the bank about any suspicious transaction.

How does credit card fraud detection help to detect and stop credit card frauds?
To detect and stop credit card fraud, a credit card company analyzes data that it receives from merchants about the consumers who have made purchases with their card. The credit card company automatically compares the data from the purchase with previously stored data on the consumer to determine whether the purchase and consumer are consistent. A computer analyzes the consumer's data and compares it with the data from the purchase. The computer also attempts to detect any difference between the consumer's history of purchases and the current purchase. The computer then makes a risk analysis for the purchase and determines whether the company should allow the purchase to go through.

What machine learning algorithm is used in credit card fraud detection?
There are several machine learning algorithms which are used in credit card fraud detection. One of the most common algorithms is SVM or support vector machines. SVM is an adaptive classification and regression algorithm with many applications in computer science. It is used in Credit card fraud detection to predict and classify a new data set into a set of predefined categories (also called classes). SVM can be used in credit card fraud detection to predict whether the new data belongs to some category which is already defined by us.

Want to share this article?
Lead the AI Driven Technological Revolution
APPLY FOR ADVANCED CERTIFICATE PROGRAMME IN MACHINE LEARNING & DEEP LEARNING
Leave a comment
Your email address will not be published. Required fields are marked *

Comment

Name *
Email *
Website

Post navigation
PREV
NEXT
Our Trending Machine Learning Courses
Advanced Certificate Programme in Machine Learning and NLP from IIIT Bangalore - Duration 8 Months
Master of Science in Machine Learning & AI from LJMU - Duration 18 Months
Executive PG Program in Machine Learning and AI from IIIT-B - Duration 12 Months
Our Popular Machine Learning Course
Machine Learning Course
Get Free Consultation
First Name
Last Name

Select Course
Email
Phone No.
City
By clicking 'Submit' you Agree to UpGrad's Terms & Conditions.
Machine Learning Skills To Master
Artificial Intelligence Courses
Tableau Courses
NLP Courses
Deep Learning Courses




SKILLS 1.Python

2.Numpy

3.Pandas

4.Matplotlib

5.Seaborn

6.feature-engine

7.Sckit-learn

8.Statistics

9.Probability

10.Deep learning

11.Machine learning




## Screenshots

![App Screenshot](https://dezyre.gumlet.io/images/blog/credit-card-fraud-detection-project-with-source-code-in-python/Credit_Card_Fraud_Detection.png?w=640&dpr=1.3)

