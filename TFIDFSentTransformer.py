

import pandas as pd
import numpy as np
import re
import string


import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from collections import Counter
from dataclasses import dataclass
from timeit import default_timer as timer
import random
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize 

import gensim
from gensim.models import Word2Vec

import spacy
from spacy import displacy

from spacy.matcher import Matcher 
from spacy.tokens import Span 

import networkx as nx
import matplotlib.pyplot as plt
from tqdm import tqdm


from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from IPython.display import display, HTML

from typing import List, Callable, Dict, Tuple, Set

pd.set_option('max_colwidth', 600)
pd.set_option('display.max_rows', 500)

#Load Sentence Transformer model optimized for  sentence cosine similarity calculations

#The models below fully downloaded in Google Colab. This is the version of the google colab notebook but 
#it was open in anoconda to be saved as pdf and the download graphics did not transfer properly so 
#it seems like it didn't download. However, it did in the orignal google colab notebook, where all the
#analysis was run.

model = SentenceTransformer('multi-qa-MiniLM-L6-cos-v1')

# Only run this once, they will be downloaded.
nltk.download('stopwords',quiet=True)
nltk.download('wordnet',quiet=True)
nltk.download('punkt',quiet=True)
nltk.download('omw-1.4',quiet=True)

#read in data
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up","hey",)
GREETING_RESPONSES = ["Hello"]


# Checking for greetings
def greeting(sentence):
    """If user's input is a greeting, return a greeting response"""
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)
import wikipedia

def download_wikipedia_article(page_title):
    try:
        page = wikipedia.page(page_title)
        return page.content
    except wikipedia.exceptions.PageError as e:
        return f"Page not found: {e}"

# Example usage
text = download_wikipedia_article("Greek mythology")
print(text[:500])  # Print first 500 characters to check

# Optionally, save to a file
with open("Greek_mythology.txt", "w", encoding="utf-8") as file:
    file.write(text)


#create list of sentences and words
sent_tokens = nltk.sent_tokenize(text)# converts to list of sentences 
word_tokens = nltk.word_tokenize(text)# converts to list of words

raw = text.lower()


sent_tokens = nltk.sent_tokenize(raw)# converts to list of sentences
word_tokens = nltk.word_tokenize(raw)# converts to list of words

# print the tokens
sent_tokens


# Generating response function 
def response(user_response):
    chatbot_response=''
    sentence_encodings=model.encode(sent_tokens, convert_to_tensor=True)# generate sentence transformer embeddings
    sentence_encodings=sentence_encodings.cpu()
    vals = cosine_similarity(sentence_encodings[-1].reshape(1, -1), sentence_encodings) #the chatbot conversation code 
    #in the next cell adds the question as the last sentence of the sentence tokens, before calling this response function.
    #The code takes the last sentence (which is the question) and gets cosine similarities vs all the sentences in the corpus,
    #including itself
    idx=vals.argsort()[0][-2] #gets the index of the second highest similarity (the first highest would be the question itself)
    flat = vals.flatten()#reduces dimension of cosine similarity array to be able to sort
    flat.sort() #sort the cosine similarity values
    second_cos_sim_val = flat[-2] #get the second highest cosine similarity value.
    if(second_cos_sim_val==0): #check the second highest cosine similarity value. If it's zero return the no match response,
        #else return highest cosine similarity sentence.
        chatbot_response=chatbot_response+"Sorry, I do not have an answer to your question in my database"
        return chatbot_response
    else:
        chatbot_response = chatbot_response+sent_tokens[idx] #use index of highest cosine similarity to get original sentence
        return chatbot_response
    
#Chatbot interaction code

flag=True
print("Welcome to the Greek mythology Chatbot. To end session please type exit")
print("\n")

while(flag==True):
    user_response = input()
    user_response=user_response.lower()
    if user_response!='exit':
        if(user_response=='thanks' or user_response=='thank you' ):
            flag=False
            print("Answer: You are welcome!")
        else:
            if(greeting(user_response)!=None):
                print("Answer: "+greeting(user_response))
            else:
                sent_tokens.append(user_response)
                word_tokens=word_tokens+nltk.word_tokenize(user_response)
                final_words=list(set(word_tokens))
                print("Answer: ",end="")
                print(response(user_response))
                print("\n")
                sent_tokens.remove(user_response)
    else:
        flag=False
        print("Thank you for using the Greek mythology Chatbot. Hope to see you soon.")    


