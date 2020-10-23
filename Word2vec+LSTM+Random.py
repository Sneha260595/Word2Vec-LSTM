#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
from PIL import Image
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
import time


# In[2]:


reviews = pd.read_csv("amazon_reviews.csv")
reviews.head().T


# In[3]:


reviews.shape


# In[4]:


pd.set_option('display.max_colwidth',1000)
reviews[['Summary','Text']].head()


# #### Checking for missing values

# In[5]:


reviews[['Summary', 'Text']].isnull().sum()


# We see that there are 27 missing values in "Summary" and no missing values in "Text". So, we are good to go

# In[6]:


fig, ax = plt.subplots()
sns.distplot(reviews['Score'], ax=ax, kde=False, color='r')

def change_width(ax, new_value) :
    for patch in ax.patches :
        current_width = patch.get_width()
        diff = current_width - new_value

        # we change the bar width
        patch.set_width(new_value)

        # we recenter the bar
        patch.set_x(patch.get_x() + diff * .5)

change_width(ax, .4)
plt.xlabel('Score')
plt.ylabel('Count of Reviews')
plt.title('Histogram of Review Scores')
plt.show()


# In[7]:


reviews['Score'].value_counts()/reviews['Score'].count()*100


# We see that ~78% of the scores are 4&5. The remaining 22% belong to 1,2&3

# In[8]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
# Define a function to plot word cloud
def plot_cloud(wordcloud):
    # Set figure size
    plt.figure(figsize=(40, 30))
    # Display image
    plt.imshow(wordcloud) 
    # No axis details
    plt.axis("off");


# In[9]:


text = reviews['Summary'][1:10000].to_string()
mask = np.array(Image.open('upvote.png'))

# Generate wordcloud
wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='white', colormap='Set2', collocations=False, stopwords=STOPWORDS, mask=mask).generate(text)

# Plot
plot_cloud(wordcloud)


# In[10]:


text = reviews['Text'][1:10000].to_string()
mask = np.array(Image.open('comment.png'))

# Generate wordcloud
wordcloud = WordCloud(width=3000, height=2000, random_state=1, background_color='white', colormap='Set2', collocations=False, stopwords=STOPWORDS, mask=mask).generate(text)

# Plot
plot_cloud(wordcloud)


# **Converting Scores 1,2,3 to 0 and 4,5 to 1 to convert the problem statement to binary classification. Removing '3' reviews as they are neutral reviews and not considering them**

# In[11]:


reviews = pd.read_csv("amazon_reviews.csv")
reviews = reviews[reviews['Score'] != 3]
reviews['Score'].replace([1,2], 0, inplace=True)
reviews['Score'].replace([4,5], 1, inplace=True)
reviews['Score'].value_counts()/reviews['Score'].count()*100


# Removing duplicate values

# In[12]:


reviews = reviews.drop_duplicates(subset={"UserId","ProfileName","Time","Text"})
reviews.shape


# **Also given that, Helpfulness Numerator should be less than or equal to Helpfulness Denominator. This means that atleast 1 person has said that this review is helpful. Hence filtering out those values where it is opposite**

# In[13]:


reviews = reviews[reviews['HelpfulnessNumerator'] <= reviews['HelpfulnessDenominator']]
reviews.shape


# ### Combing "Summary" and "Text" which will serve as the complete text for our modeling

# In[14]:


reviews['Text_Clean'] = reviews['Summary'].str.cat(reviews['Text'], sep =". ")
reviews[['Summary', 'Text', 'Text_Clean']].head()


# The Text_Clean is now a combination of both Summary and Text

# In[15]:


reviews = reviews[['Score', 'Text_Clean']]
reviews = reviews.drop([33958])
reviews.reset_index(inplace=True)


# Our data set is ready for modeling purpose

# ## Data Cleaning

# **Stemming - Stopwords**

# In[16]:


nltk.download('stopwords')


# In[17]:


reviews=reviews[['Score','Text_Clean']]
reviews.head()

In order to balance the data we have taken 50000 rows from each positive and negative reviews
# In[18]:


reviews_pos = reviews[reviews.Score==1][:50000]
reviews_neg = reviews[reviews.Score==0][:50000]
reviews_new = reviews_pos.append(reviews_neg)
reviews_new.head()


# In order to proceed in any NLP pipeline you will have to tokenize the data  i.e, by breaking down paras into sentences into words.

# In[19]:


reviews_new.shape
reviews_new = reviews_new.sample(frac = 1).reset_index(drop = True)
reviews_new.head()


# In[20]:


from bs4 import BeautifulSoup
from nltk.stem import WordNetLemmatizer
def clean_reviews(review):  
 
    review_text = BeautifulSoup(review,"lxml").get_text()
    review_text = re.sub("[^a-zA-Z]"," ",review_text)
    word_tokens= review_text.lower().split()
    le=WordNetLemmatizer()
    stop_words= set(stopwords.words("english"))     
    word_tokens= [le.lemmatize(w) for w in word_tokens if not w in stop_words] 
    cleaned_review=" ".join(word_tokens)
    return cleaned_review

tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
sentences= []
sum = 0
for review in reviews_new['Text_Clean']:
    sents= tokenizer.tokenize(review.strip())
    sum+=len(sents)
    for sent in sents:
        cleaned_sent=clean_reviews(sent)
        sentences.append(cleaned_sent.split()) 

    


# In[21]:


print(len(sentences))
for sentence in sentences[:10]:
    print(sentence,"\n")


# Wod2vec embeddings can be done as follows:
# 

# In[22]:


import gensim
w2v_model = gensim.models.Word2Vec(sentences=sentences,size=400,window=10,min_count=1)


# In[24]:


w2v_model.train(sentences,epochs=10,total_examples=len(sentences))


# In[23]:


w2v_model.wv.get_vector('nice')


# In[25]:


vocab=w2v_model.wv.vocab
print("Total number of words", len(vocab))


# In[26]:


w2v_model.wv.most_similar('nice')


# Similarity between two words

# In[27]:


w2v_model.wv.similarity('nice','good')


# Now creating a dictionary with words in vocab and their embeddings

# In[28]:


vocab = list(vocab.keys())


# In[29]:


word2vec_dict={}
for word in vocab:
    word2vec_dict[word]=w2v_model.wv.get_vector(word)
    
print("Total number of key-value pairs: ", len(word2vec_dict))


# In[30]:


for word in vocab[:2]:
    print(word2vec_dict[word],"\n")


# Building classifier using word embedding :
# 

# In[31]:


reviews_new['clean_Review']= reviews_new['Text_Clean'].apply(clean_reviews)
reviews_new.head()


# Data for Keras Embedding layer 

# In[32]:


maxi=-1
for i,rev in enumerate(reviews_new['clean_Review']):
  tokens=rev.split()
  if(len(tokens)>maxi):
    maxi=len(tokens)
print(maxi)


# In[33]:



import keras
from keras.preprocessing.text import one_hot,Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense , Flatten ,Embedding,Input,LSTM
from keras.models import Model
from keras.preprocessing.text import text_to_word_sequence
tok = Tokenizer()
tok.fit_on_texts(reviews_new['clean_Review'])
vocab_size = len(tok.word_index) + 1
encd_rev = tok.texts_to_sequences(reviews_new['clean_Review'])


# In[34]:


max_rev_len=1570  
vocab_size = len(tok.word_index) + 1  
embed_dim=400
pad_rev= pad_sequences(encd_rev, maxlen=max_rev_len, padding='post')
pad_rev.shape


#     Creating Embedded Matrix

# In[35]:


embed_matrix=np.zeros(shape=(vocab_size,embed_dim))
for word,i in tok.word_index.items():
  embed_vector=word2vec_dict.get(word)
  if embed_vector is not None:  
    embed_matrix[i]=embed_vector

    
print(embed_matrix[14])


# Training and validation Sets creation:

# In[36]:


from sklearn.model_selection import train_test_split
Y=keras.utils.to_categorical(reviews_new['Score'])  
x_train,x_test,y_train,y_test=train_test_split(pad_rev,Y,test_size=0.20,random_state=42)


# In[39]:


from keras.initializers import Constant
from keras.layers import ReLU
from keras.layers import Dropout

model=Sequential()
model.add(Embedding(input_dim=vocab_size,output_dim=embed_dim,input_length=max_rev_len,embeddings_initializer=Constant(embed_matrix)))
model.add(LSTM(64,return_sequences=False))
model.add(Flatten())
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.50))
model.add(Dense(16,activation='relu'))
model.add(Dropout(0.20))
model.add(Dense(2,activation='sigmoid'))


# In[64]:


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators = 100)

print("Fitting a random forest to labeled training data...")
forest = forest.fit(x_train, y_train)


# In[40]:


model.summary()


# In[74]:


from sklearn import metrics
import numpy as np
import itertools
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
pred = forest.predict(x_test)
score = metrics.accuracy_score(y_test, pred)


# In[41]:


model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-3),loss='binary_crossentropy',metrics=['accuracy'])


# In[71]:


print(score)


# In[42]:


epochs=5
batch_size=64


# In[73]:


print("Accuracy:",metrics.accuracy_score(y_test, pred))


# In[49]:


model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data=(x_test,y_test))


# In[48]:


from sklearn.metrics import classification_report
pred = np.argmax(model.predict(x_test), axis=1)
y_true = np.argmax(y_test, axis=1)
print(classification_report(y_true, pred))


# In[80]:


print(classification_report(y_test,pred))


# In[96]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    See full source and example:
    http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
   
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="left",
                 color="red" if cm[i, j] > thresh else "red")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

cm = metrics.confusion_matrix(y_test.argmax(axis=1), pred.argmax(axis=1))
plot_confusion_matrix(cm, classes=['NEGATIVE', 'POSITIVE'])


# In[ ]:




