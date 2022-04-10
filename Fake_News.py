#Made by kaioont
#This program i made detects real or fake news 
#----------------------------------------------#
#importing libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import cross_val_score
import pandas as pd
#Loading the data and configuring it 
df = pd.read_csv('News/train.csv')
conversation_dict = {0: 'Real', 1: 'Fake'}
df['label'] = df['label'].replace(conversation_dict)
#This part is vectorizeing the text and telling you the accuracy of the program 
x_train,x_test,y_train,y_test=train_test_split(df['text'], df['label'], test_size=0.25, random_state=7, shuffle=True)
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.75)
vec_train=tfidf_vectorizer.fit_transform(x_train.values.astype('U'))
vec_test=tfidf_vectorizer.transform(x_test.values.astype('U'))
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(vec_train,y_train)
y_pred=pac.predict(vec_test)
score=accuracy_score(y_test,y_pred)
print(f'PAC Accuracy: {round(score*100,2)}%')
confusion_matrix(y_test,y_pred, labels=['Real','Fake'])
X=tfidf_vectorizer.transform(df['text'].values.astype('U'))
scores = cross_val_score(pac, X, df['label'].values, cv=5)
print(f'K Fold Accuracy: {round(scores.mean()*100,2)}%')
#this will load the test data 
df_true=pd.read_csv('Test/True.csv')
df_true['label']='Real'
df_true_rep=[df_true['text'][i].replace('WASHINGTON (Reuters) - ','').replace('LONDON (Reuters) - ','').replace('(Reuters) - ','') for i in range(len(df_true['text']))]
df_true['text']=df_true_rep
df_fake=pd.read_csv('Test/Fake.csv')
df_fake['label']='Fake'
df_final=pd.concat([df_true,df_fake])
df_final=df_final.drop(['subject','date'], axis=1)
df_fake
#idk anymore its just a def so ya 
def findlabel(newtext):
 vec_newtest=tfidf_vectorizer.transform([newtext])
 y_pred1=pac.predict(vec_newtest)
 return y_pred1[0]
