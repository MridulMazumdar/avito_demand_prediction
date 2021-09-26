import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import re

from bs4 import BeautifulSoup
import nltk

from nltk.corpus import stopwords
from string import punctuation
from tqdm.notebook import tqdm_notebook as tqn
from simplemma import text_lemmatizer
import simplemma
import cv2
from tensorflow import keras 
from tensorflow.keras import backend as kb
import pickle
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence

nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")
langdata = simplemma.load_data('ru')



def process_data(df):
    
    tqn.pandas()

    df['description']=df['description'].fillna('')
    df['image']=df['image'].fillna('no img')
    df['price']=df['price'].fillna(0)
    def log_price(x):
        if x!=0:
            return np.log(x)
        else:
            return x
    
    df['price']=df['price'].apply(log_price)
    
    df[['param_1','param_2','param_3']]=df[['param_1','param_2','param_3']].fillna('')
    df['full_text']=df['param_1']+df['param_2']+df['param_3']+df['title']+df['description']
    #date_features
    df['activation_date']=pd.to_datetime(df['activation_date'])
    df['act_month']=df['activation_date'].dt.month
    df['act_day']=df['activation_date'].dt.day
    df['act_dow']=df['activation_date'].dt.dayofweek
    df['dayofweek_name'] = df['activation_date'].dt.day_name
    df['is_weekend'] = np.where(df['dayofweek_name'].isin(['Sunday','Saturday']),1,0)
    
    del df['activation_date'],df['dayofweek_name']
    
    
    def get_blurrness_score(image):
        images_path = 'data/images/'

        try:
            path =  images_path + image +'.jpg' 
            image = cv2.imread(path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            fm = cv2.Laplacian(image, cv2.CV_64F).var()
            return fm
        except:
            return 0
    


    def string_process(sentance):
        sentance = re.sub(r"http\S+", "", sentance)
        sentance = BeautifulSoup(sentance, 'lxml').get_text()
        sentance = re.sub('\W+',' ', sentance )
        sentance = re.sub('_','',sentance)
  
        sentance=' '.join(text_lemmatizer(sentance, langdata))
        
        sentance =' '.join([word for word in sentance.split() if word not in (russian_stopwords)])
        
        sentance =' '.join([word for word in sentance.split() if word not in (punctuation)])
        
        return sentance.lower()
    
    df['full_text']=df['full_text'].progress_apply(string_process)
    
    df['char_count'] = df['description'].progress_apply(len)
    df['word_count'] = df['description'].progress_apply(lambda x: len(x.split()))
    df['word_density'] = df['char_count'] / (df['word_count']+1)
    df['punctuation_count'] = df['description'].progress_apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 
    df['title_word_count'] = df['title'].progress_apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    
    del df['title'],df['description'],df['param_1'],df['param_2'],df['param_3']
    
    cat_feat=['user_type','region','city','parent_category_name','category_name']
    for i in tqn(cat_feat):
        df[i]=df[i].str.lower()
        encoder = LabelEncoder()
        encoder.classes_=np.load(i+'classes.npy',allow_pickle=True)
        
        df[i] = df[i].map(lambda s:'<unknown>' if s not in encoder.classes_ else s)
        encoder.classes_ = np.append(encoder.classes_, '<unknown>')
        
        df[i]=encoder.transform(df[i].str.lower())
    
    df['blurrness'] = df['image'].progress_apply(get_blurrness_score)

    return df


def deal_prob(df):
    drop_cols=['item_id','user_id','image','image_top_1','item_seq_number']

    cat_cols=['region','city','parent_category_name','category_name','user_type','is_weekend','act_month','act_day',
          'act_dow']

    num_cols=['price','blurrness','char_count','word_count','word_density','punctuation_count',
          'title_word_count']

    target='deal_probability'
    
    df=df.drop(drop_cols,axis=1)
    
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
        
    final_essay=sequence.pad_sequences(tokenizer.texts_to_sequences(df['full_text']),maxlen=400)
    del df['full_text']
    
    def tweedieloss(y_true, y_pred):
        p=1.1
        a = y_true*kb.pow(y_pred, (1-p)) / (1-p)
        b = kb.pow(y_pred, (2-p))/(2-p)
        loss = -a + b
        return loss 

    def rmse(y_true,y_pred):
        return kb.sqrt(kb.mean(kb.square(y_pred-y_true)))
    
    model=keras.models.load_model('model1_best3.h5',custom_objects={'rmse':rmse}) 
    
    df[target]=model.predict([final_essay,
                              df[cat_cols],
                              df[num_cols]])
    
    return df[target]

df=pd.read_csv('test.csv')

pro_df=process_data(df)

deal_prob(pro_df)