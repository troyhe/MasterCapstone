import numpy as np
import pandas as pd


from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn import tree
from sklearn import svm


#create input dataframe
def split_input(path):
    splitted = pd.DataFrame()
    temp = pd.read_csv(path,sep='\n',names=['string'])
    #print(temp)
    output = temp['string'].str.split(":::",expand=True,n=1)
    splitted['text'] = output[1]
    #print(output[0])
    meta = output[0].str.split(',',expand=True,n=3)
    meta_cols = ['RT','emoji','URL','mention']
    for idx,atr in enumerate(meta_cols):
        splitted[atr] = meta[idx].str.split(':',expand=True,n=1)[1]
    splitted['mention'] = splitted['mention'].str.strip('}')
    return splitted

#Read in the data
def create_input(path,File):
    df = pd.DataFrame()
    for file in File:
        if(file == 'bot.txt'):
            temp = split_input(path+file)
            temp['label'] = 1
            df = pd.concat([df,temp])
        else:
            temp = split_input(path+file)
            temp['label'] = 0
            df = pd.concat([df,temp])
    return df

path = 'dataset-light-3000/'
File=['human_male.txt','bot.txt']

train = create_input(path+'train/',File)
validate = create_input(path+'test/',File)

train.info()
validate.info()

#vectorize the text data
# vectorizer = CountVectorizer(stop_words='english')
vectorizer = CountVectorizer(analyzer='char',ngram_range=(5,5),min_df=3)
all_text = pd.concat([train['text'],validate['text']])
vectorizer.fit(all_text)
train['text'] = vectorizer.transform(train['text']).toarray()
validate['text'] = vectorizer.transform(validate['text']).toarray()

#train models with metadata
X = train[['text','RT','emoji','URL','mention']]
random_forest_m = RandomForestClassifier(bootstrap=True,max_depth=5).fit(X,train['label'])

T_Y = random_forest_m.predict(X)
print(f1_score(train['label'],T_Y))

#validate models with metadata
V_X = validate[['text','RT','emoji','URL','mention']]
V_Y = random_forest_m.predict(V_X)
print(f1_score(validate['label'],V_Y))



