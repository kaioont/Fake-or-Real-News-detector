# these are the instructions for how to change the test data  
### findlabel((df_true['text'][0]) this will take a article from the test folder it is set to get the first article from the True csv file you can change this bye changeing the df_true to df_fake this will change this to the fake csv file you
# how to see the accuracy 
### the program automatically shows the PAC Accuracy and the K Fold Accuracy but to get a more accurate reading just run sum([1 if findlabel((df_true['text'][i]))=='Real' else 0 for i in range(len(df_true['text']))])/df_true['text'].size
# run in interactive window

Data to Build Model: https://www.kaggle.com/c/fake-news/data

Data to Test Model: https://www.kaggle.com/clmentbisaillon/fake-and-real-news-dataset
