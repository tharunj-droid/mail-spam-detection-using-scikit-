import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB , GaussianNB
from sklearn import svm
from sklearn.model_selection import GridSearchCV
#load the data set
dataframe = pd.read_csv("D:\Tharun\own codes\spam_detection\spam.csv")
print(dataframe.head())
#print(dataframe.describe())
#split for training and testing
x=dataframe["EmailText"]
y= dataframe["Label"]
#splitting 80 20 80 to train and 20 to test
x_train,y_train=x[0:4457],y[0:4457]
x_test,y_test=x[4457:],y[4457:]
#extarcting features
cv = CountVectorizer()
features = cv.fit_transform(x_train)
#build a model
tuned_parameters={'kernel':['linear','rbf'],'gamma':[1e-3,1e-4],'C':[1,10,100,1000]}
model = GridSearchCV(svm.SVC(),tuned_parameters)
model.fit(features,y_train)
#testing accuracy
print(model.best_params_)
#Step5: Test Accuracy
print("the accuracy of the model is",model.score(cv.transform(x_test),y_test))
