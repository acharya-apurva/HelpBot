import numpy as np
import nltk
#nltk.download('punkt') #package with a pre-trained tokenizer
from nltk.stem.porter import PorterStemmer #importing a stemmer (there are different stemmers available)

stemmer= PorterStemmer()

def tokenize(sentence):
    return nltk.word_tokenize(sentence)
def stem(word):
    return stemmer.stem(word.lower())
def bagOfWords(tSentence, allWords): #array of the words in a particular sentence and array of all words passed as parameters; for 
#word in tSentence, put one at the appropriate position for allWords
    tSentence = [stem(w) for w in tSentence]
    bag = np.zeros(len(allWords), dtype=np.float32)#array of zeroes of len all words
    for i, w in enumerate(allWords): #enumerate pairs each item with a function
        if w in tSentence:
            bag[i] = 1.0 #if it contains the word turn the zero into one
    return bag

#testing bagging
#sentence =["What", "do", "you", "want"]
#allWords =["my","what","who", "do", "you", "want"]
#bag = bagOfWords(sentence,allWords)
#print(bag)


#testing tokenizing
#a = "What is your name?"
#a= tokenize(a)
#print(a)

#code for testing stemming
#words= ["dance","dancing","danced"]
#stemmedWords = [stem(w) for w in words]
#print(stemmedWords)

