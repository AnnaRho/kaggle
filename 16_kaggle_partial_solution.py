'''
CLASS: Kaggle Stack Overflow competition
'''

# read in the file and set the first column as the index
import pandas as pd
train = pd.read_csv('../../assets/dataset/train.zip', index_col=0)
train.head(2)
train.shape

'''
What are some assumptions and theories to test?

OwnerUserId: not unique within the dataset, assigned in order
OwnerCreationDate: users with older accounts have more open questions
ReputationAtPostCreation: higher reputation users have more open questions
OwnerUndeletedAnswerCountAtPostTime: users with more answers have more open questions
Title and BodyMarkdown: well-written questions are more likely to be open
Tags: 1 to 5 tags are required, many unique tags
OpenStatus: most questions should be open (encoded as 1)
'''

## OPEN STATUS

# dataset is perfectly balanced in terms of OpenStatus (not a representative sample)
train.OpenStatus.value_counts()
set(train.OpenStatus)

#set as a binary classification problem
train.OpenStatus = train.OpenStatus.map({'open':1, 
                                        'not a real question':0,
                                        'not constructive':0,
                                        'off topic':0,
                                        'too localized':0})
## USER ID

# OwnerUserId is not unique within the dataset, let's examine the top user
train.OwnerUserId.value_counts()

# mostly closed questions, few answers, all lowercase, grammatical mistakes
train[train.OwnerUserId==466534].head(10)

# let's find a user with a high proportion of open questions
train.groupby('OwnerUserId').OpenStatus.mean()
train.groupby('OwnerUserId').OpenStatus.agg(['mean','count']).sort_values('count')

# lots of answers, better grammar, multiple tags, all .net
train[train.OwnerUserId==185593].head(10)

#set pandas display options to view long text 
pd.set_option('display.width', 50)
pd.set_option('display.max_colwidth', 100)

#opposite case, many closed questions
train[train.OwnerUserId==466534].head(10)

## REPUTATION

# ReputationAtPostCreation is higher for open questions: possibly use as a feature
train.groupby('OpenStatus').ReputationAtPostCreation.describe().unstack()

# not a useful histogram
train.ReputationAtPostCreation.plot(kind='hist')

# much more useful histogram
train[train.ReputationAtPostCreation < 1000].ReputationAtPostCreation.plot(kind='hist')

# grouped histogram
train[train.ReputationAtPostCreation < 1000].hist(column='ReputationAtPostCreation', 
                                                    by='OpenStatus', sharey=True, figsize=(10, 8))

# grouped box plot
train[train.ReputationAtPostCreation < 1000].boxplot(column='ReputationAtPostCreation',
                                                         by='OpenStatus', figsize=(10, 8))

## ANSWER COUNT

# rename column
train.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)

# Answers is higher for open questions: possibly use as a feature
train.groupby('OpenStatus').Answers.describe().unstack()


## TITLE LENGTH

# create a new feature that represents the length of the title (in characters)
train['TitleLength'] = train.Title.apply(len)

# Title is longer for open questions: possibly use as a feature
train.groupby('OpenStatus').TitleLength.describe().unstack()
train.boxplot(column='TitleLength', by='OpenStatus')


'''
Define a function that takes a raw CSV file and returns a DataFrame that
includes all created features (and any other modifications)
'''

# define the function
def make_features(filename):
    df = pd.read_csv(filename, index_col=0)
    
    if 'train' in filename:
        df.OpenStatus = df.OpenStatus.map({'open':1, 
                                        'not a real question':0,
                                        'not constructive':0,
                                        'off topic':0,
                                        'too localized':0})
    if 'test' in filename:
        df.drop('PostClosedDate', axis=1, inplace=True)
            
    df.rename(columns={'OwnerUndeletedAnswerCountAtPostTime':'Answers'}, inplace=True)
    df['TitleLength'] = df.Title.apply(len)
    return df

# apply function to both training and testing files
train = make_features('../../assets/dataset/train.zip')
test = make_features('../../assets/dataset/test.zip')


'''
Evaluate a model with three features
'''

# define X and y
feature_cols = ['ReputationAtPostCreation', 'Answers', 'TitleLength']
X = train[feature_cols]
y = train.OpenStatus

# split into training and testing sets
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

# fit a logistic regression model
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(C=1e9)
logreg.fit(X_train, y_train)

# examine the coefficients to check that they makes sense
logreg.coef_

# predict response classes and predict class probabilities
y_pred_class = logreg.predict(X_test)
y_pred_prob = logreg.predict_proba(X_test)[:, 1]

# check how well we did
from sklearn import metrics
metrics.accuracy_score(y_test, y_pred_class)    # 0.562 (better than guessing)
metrics.confusion_matrix(y_test, y_pred_class)  # predicts closed a lot of the time
metrics.roc_auc_score(y_test, y_pred_prob)      # 0.589 (not horrible)
metrics.log_loss(y_test, y_pred_prob)           # 0.684 (what is this?)

# let's see if cross-validation gives us similar results
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(logreg, X, y, scoring='log_loss', cv=10)
scores.mean()       # 0.684 (identical to train/test split)
scores.std()        # very small


'''
Understanding log loss
'''

# 5 pretend response values
y_test = [0, 0, 0, 1, 1]

# 5 sets of predicted probabilities for those observations
y_pred_prob_sets = [[0.1, 0.2, 0.3, 0.8, 0.9],
                    [0.4, 0.4, 0.4, 0.6, 0.6],
                    [0.4, 0.4, 0.7, 0.6, 0.6],
                    [0.4, 0.4, 0.9, 0.6, 0.6],
                    [0.5, 0.5, 0.5, 0.5, 0.5]]

# calculate AUC for each set of predicted probabilities
for y_pred_prob in y_pred_prob_sets:
    print(y_pred_prob, metrics.roc_auc_score(y_test, y_pred_prob))

# calculate log loss for each set of predicted probabilities
for y_pred_prob in y_pred_prob_sets:
    print(y_pred_prob, metrics.log_loss(y_test, y_pred_prob))


'''
Create a submission file
'''

# train the model on ALL data (not X_train and y_train)
logreg.fit(X, y)

# predict class probabilities for the actual testing data (not X_test)
X_oos = test[feature_cols]
oos_pred_prob = logreg.predict_proba(X_oos)[:, 1]

# sample submission file indicates we need two columns: PostId and predicted probability
test.index      # PostId
oos_pred_prob   # predicted probability

# create a DataFrame that has 'id' as the index, then export to a CSV file
sub = pd.DataFrame({'id':test.index, 'OpenStatus':oos_pred_prob}).set_index('id')
sub.to_csv('sub1.csv')  # 0.687
