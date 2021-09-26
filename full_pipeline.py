import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers,Input,callbacks,losses,optimizers
import numpy as np
import pandas as pd
from tensorflow.keras import backend as kb
from tensorflow.keras.preprocessing.text import Tokenizer,text_to_word_sequence
from tensorflow.keras import callbacks
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from tqdm.notebook import tqdm_notebook as tqn
from datetime import datetime
import pickle
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
from PIL import Image as IMG
from skimage import feature

nltk.download("stopwords")
russian_stopwords = stopwords.words("russian")
langdata = simplemma.load_data('ru')


df=pd.read_csv('train.csv')


def process_train_data(df):
    
    tqn.pandas()

    df['description']=df['description'].fillna('')
    df['image']=df['image'].fillna('no img')
    df[['param_1','param_2','param_3']]=df[['param_1','param_2','param_3']].fillna('')
    df['full_text']=df['param_1']+df['param_2']+df['param_3']+df['title']+df['description']
    #date_features
    df['activation_date']=pd.to_datetime(df['activation_date'])
    df['act_month']=df['activation_date'].dt.month
    df['act_day']=df['activation_date'].dt.day
    df['act_dow']=df['activation_date'].dt.dayofweek
    df['dayofweek_name'] = df['activation_date'].dt.day_name
    df['is_weekend'] = np.where(df['dayofweek_name'].isin(['Sunday','Saturday']),1,0)
    
    df['price']=df['price'].fillna(0)
    def log_price(x):
        if x!=0:
            return np.log(x)
        else:
            return x
    
    df['price']=df['price'].apply(log_price)
    
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
    
    #adding features based on 
    df['char_count'] = df['description'].progress_apply(len)
    df['word_count'] = df['description'].progress_apply(lambda x: len(x.split()))
    df['word_density'] = df['char_count'] / (df['word_count']+1)
    df['punctuation_count'] = df['description'].progress_apply(lambda x: len("".join(_ for _ in x if _ in punctuation))) 
    df['title_word_count'] = df['title'].progress_apply(lambda x: len([wrd for wrd in x.split() if wrd.istitle()]))
    
    del df['title'],df['description'],df['param_1'],df['param_2'],df['param_3']
    
    cat_feat=['user_type','region','city','parent_category_name','category_name']
    for i in tqn(cat_feat):
        encoder = LabelEncoder()
        encoder.fit(df[i].str.lower())
        np.save(i+'classes.npy', encoder.classes_)
        df[i]=encoder.transform(df[i].str.lower())
    
    df['blurrness'] = df['image'].progress_apply(get_blurrness_score)

    return df



def model_train(df):
    drop_cols=['item_id','user_id','image','image_top_1','item_seq_number']

    cat_cols=['region','city','parent_category_name','category_name','user_type','is_weekend','act_month','act_day',
          'act_dow']

    num_cols=['price','blurrness','char_count','word_count','word_density','punctuation_count',
          'title_word_count']

    target='deal_probability'

    df=df.drop(drop_cols,axis=1)
    X=df.drop(target,axis=1)
    Y=df[target].astype('float32')
    
    x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.03)

    del df
    
    embedding_file='cc.ru.300.vec'

    embeddings_index = {}
    with open(embedding_file,encoding='utf8') as f:
        for line in tqn(f):
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float16')
            embeddings_index[word] = coefs
        
    f.close()
    
    def embedding_matrix(word_index):
        embedding_matrix = np.zeros((len(word_index)+1, 300))
        for word, i in tqn(word_index.items()):
            embedding_vector = embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector

        return embedding_matrix
    
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(x_train['full_text'])

    with open('/content/drive/MyDrive/AppliedAICourse/Self_Case_Study_2/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        
    
    train_final_essay_p1=sequence.pad_sequences(tokenizer.texts_to_sequences(x_train['full_text']),maxlen=400)
    test_final_essay=sequence.pad_sequences(tokenizer.texts_to_sequences(x_test['full_text']),maxlen=400)

    full_text_matrix=embedding_matrix(tokenizer.word_index)
    
    def embedding_layer(word_index,weight_matrix):
        return layers.Embedding(len(word_index)+1,
                            300,
                            weights=[weight_matrix],
                            trainable=False)

    I1 = Input(shape=(400,),name='I1')  
    embedding1=embedding_layer(tokenizer.word_index,full_text_matrix)(I1)
    lstm=layers.LSTM(40)(embedding1)
    flatten=layers.Flatten()(lstm)

    I2= Input(shape=(len(cat_cols),),name='I2')

    I3= Input(shape=(len(num_cols),),name='I3')
    
    concat1=layers.concatenate([I2,I3,flatten])

    BN=layers.BatchNormalization()(concat1)

    D=layers.Dense(128,activation='relu',kernel_initializer='he_uniform')(BN)

    D=layers.Dropout(0.4)(D)
    
    D=layers.Dense(64,activation='relu',kernel_initializer='he_uniform')(D)

    pred=layers.Dense(1,activation='sigmoid')(D)

    model1=keras.Model(inputs=[I1,I2,I3],outputs=pred)
    
    def rmse(y_true,y_pred):
        return kb.sqrt(kb.mean(kb.square(y_pred-y_true)))

    model1.compile(optimizer=optimizers.Adam(learning_rate=0.01),
               loss=losses.Poisson(),
               metrics=rmse)
    
    early_stop=callbacks.EarlyStopping(monitor='val_rmse',mode='min',patience=5)

    model_lib='model1_best3.h5' #poisson

    model_checkpoint=callbacks.ModelCheckpoint(model_lib,monitor='val_rmse',save_best_only=True,mode='min')
    log_dir='Model1/'+datetime.now().strftime("%Y%m%d-%H%M%S")

    model1.fit([train_final_essay_p1,
            x_train[cat_cols],
            x_train[num_cols]],
           y_train,
            batch_size=500,
           epochs=100,
          callbacks=[early_stop,model_checkpoint],
          validation_data=([test_final_essay,
                            x_test[cat_cols],
                            x_test[num_cols]],y_test))
    
    model1=keras.models.load_model('model1_best3.h5',custom_objects={'rmse':rmse})
    
    return model1

final_df=process_train_data(df)
model_train(final_df)
