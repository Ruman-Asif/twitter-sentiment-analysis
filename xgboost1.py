import numpy as np
import pandas as pd
import xgboost as xgb

# Load your data into a pandas DataFrame or read from a file
df=pd.read_csv("C:\\Users\\Ruman Asif\\Documents\\Personal\\guvi\\"
               "final project\\twitter_new.csv",encoding='ISO-8859-1'
               ,names=['Polarity','ids','date','flag','user','text'])

df.drop(['flag'], axis=1,inplace=True)

df1 = df[:150000]   #taking top 100000 rows
df2=df.tail(150000)   #taking bottom 100000 rows
df = pd.concat([df1, df2], ignore_index=True)   #concating both

#---------count vectorizer here----------------------

from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import RegexpTokenizer
token=RegexpTokenizer(r'[a-zA-Z0-9]+')
cv=CountVectorizer(stop_words='english',ngram_range=(1,3),tokenizer=token.tokenize)
text_counts=cv.fit_transform(df['text'])

#----------------------------------------------------------

#-------------assigning X and y here ----------------------
#note: X is got from countvectorizer above in the form of textcounts---------

# Split data into X (features) and y (target variable)
# X = df['text','user']
y = df['Polarity']



from sklearn.model_selection import train_test_split

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(text_counts, y , test_size=0.2) #, random_state=42)

#--------------------------using xgboost-------------------------------------

# Create an XGBoost classifier object
model = xgb.XGBClassifier()

# Fit the model on the training data
model.fit(X_train, y_train)

# Make predictions on the test data
predictions = model.predict(X_test)

from sklearn.metrics import accuracy_score

# Calculate accuracy score
accuracy = accuracy_score(y_test, predictions)
print("Accuracy of our model through xgboost is:",accuracy)

#-------------------------------------------------------------------------------

#---------------------using decision tree--------------------------------------

# from sklearn.tree import DecisionTreeClassifier
# dt=DecisionTreeClassifier()
# dt.fit(X_train, y_train)
#
# from sklearn import metrics
# predicted=dt.predict(X_test)
# # print(predicted)
# accuracy_score=metrics.accuracy_score(predicted,y_test)
# print("accuracy:",accuracy_score)

#-------------------------------------------------------------------------------

#----------------testing---------------------
# dt.predict("How are you doing?")
# f=["How are you doing?","bye"]
# # print(type(f))
# # input=[input]
# k=cv.fit_transform(df['text'])
# print(dt.predict(cv.fit_transform(f)))

#--------------------------------------------------

#------------for predictions here----------------------
f="I am sad"
f_counts=cv.transform([f])
print("The polarity for given sentence:",f,"is:",model.predict(f_counts))
print("0 is negative and 4 is positive")

#---------------------------------------------------
