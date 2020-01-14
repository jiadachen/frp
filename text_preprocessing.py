# -*- coding: utf-8 -*-
"""
Created on 
@author: 
"""

import os
import re
import csv
import glob
import nltk
import string
import numpy as np
import pandas as pd

from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def main():

	lemmatizer = WordNetLemmatizer()


	# Importing files
	## text corpus
	file_list = glob.glob(os.path.join(os.getcwd(), "files", "10k", "*.txt"))
	corpus = []
	file_dates = []

	for file_path in file_list:
	    file_dates.append(re.search(r'\d{4}-\d{2}-\d{2}', file_path)[0][:4])
	    with open(file_path) as f_input:
	        corpus.append(f_input.read())

	### group corpus based on year
	df = pd.DataFrame({'Years' : file_dates, 'corpus' : corpus})
	corpus_dict = df.groupby('Years')['corpus'].apply(lambda g: g.values.tolist()).to_dict()

	## extra stop words
	file_list2 = glob.glob(os.path.join(os.getcwd(), "files", "extra_stop_words", "*.txt"))
	extra_stop_words_corpus = []

	for file_path1 in file_list2:
	    with open(file_path1) as f_input1:
	        extra_stop_words_corpus.append(f_input1.read())

	names_stop_words = extra_stop_words_corpus[0]
	geographic_stop_words = extra_stop_words_corpus[1]
	generic_stop_words = extra_stop_words_corpus[3]
	dates_stop_words = extra_stop_words_corpus[2]


	# Preprocessing helper functions
	## Get all the text to lowercase:
	def text_lowercase(text):
	    return text.lower()

	## Remove numbers 
	def remove_numbers(text): 
	    result = re.sub(r'\d+', '', text) 
	    return result 

	## Remove punctuation
	def remove_punctuation(text):
	    translator = str.maketrans('', '', string.punctuation)
	    return text.translate(translator)

	## Remove whitespace from text
	def remove_whitespace(text):
	    return " ".join(text.split())

	## Get word type for each word:
	def get_wordnet_pos(word):
	    """Map POS tag to first character lemmatize() accepts"""
	    tag = nltk.pos_tag([word])[0][1][0].upper()
	    tag_dict = {"J": wordnet.ADJ,
	                "N": wordnet.NOUN,
	                "V": wordnet.VERB,
	                "R": wordnet.ADV}
	    return tag_dict.get(tag, wordnet.NOUN)

	## Lemmatize string
	def lemmatize_word(text):
	    word_tokens = word_tokenize(text)
	    # provide context i.e. part-of-speech
	    lemmas = [lemmatizer.lemmatize(word, pos =get_wordnet_pos(word)) for word in word_tokens]
	    return ' '.join(lemmas)

	## Remove numeric of words:
	def remove_lastfirst_numeric(txt):
	    for j, i in enumerate(txt):
	        if i[-1].isnumeric():
	            txt[j] = i[:-1]
	        if i[0].isnumeric():
	            txt[j] = i[1:]
	    return txt


	## Remove stopwords
	def remove_stopwords(text):
	    stop_words = set(full_stop_words_list)
	    word_tokens = word_tokenize(text)
	    filtered_text = [word for word in word_tokens if word not in stop_words]
	    return filtered_text

	## master function of preprocerssing text files
	def text_preprocessing(text):
	    text = text_lowercase(text)
	    text = remove_numbers(text)
	    text = remove_punctuation(text)
	    text = remove_whitespace(text)
	    text = lemmatize_word(text)
	    text = remove_stopwords(text)
	    return remove_lastfirst_numeric(text)

	# Create a final list of stop words
	## first clean up withe extra stop words list
	def clean_extra_stop_words(extra_stop_words):
	    extra_stop_words = text_lowercase(extra_stop_words)
	    extra_stop_words = remove_numbers(extra_stop_words)
	    extra_stop_words = remove_punctuation(extra_stop_words)
	    extra_stop_words = remove_whitespace(extra_stop_words)
	    return list(extra_stop_words.split())

	names_stop_words_clean = clean_extra_stop_words(names_stop_words)
	geographic_stop_words_clean = clean_extra_stop_words(geographic_stop_words)
	generic_stop_words_clean = clean_extra_stop_words(generic_stop_words)
	dates_stop_words_clean = clean_extra_stop_words(dates_stop_words)

	## union nltk stopwords list with extra stop words
	full_stop_words_list = list(set(stopwords.words("english") + names_stop_words_clean + geographic_stop_words_clean + \
	         generic_stop_words_clean + dates_stop_words_clean))


	# 10K file section parsing helper functions

	## 1A Risk factors function 
	def section_1a(corpus):
	    trimmed_corpus = []
	    for text in corpus:
	        a = remove_whitespace(text_lowercase(text))
	        start_idx = a.find('risk factors')
	        end_idx = a.find('unresolved staff comments')
	        if start_idx != -1:
	            if end_idx != -1:
	                trimmed_corpus.append(a[start_idx:end_idx])
	            elif a.find('item 2') != -1:
	                trimmed_corpus.append(a[start_idx:a.find('item 2') ])
	            elif a.find('properties') != -1:
	                trimmed_corpus.append(a[start_idx:a.find('properties') ])
	    return trimmed_corpus
	        

	## 1 Business description function
	def section_business(corpus):
	    trimmed_corpus = []
	    for text in corpus:
	        a = remove_whitespace(text_lowercase(text))
	        start_idx = 0
	        end_idx = a.find('risk factors')
	        if end_idx != -1:
	            trimmed_corpus.append(a[start_idx:end_idx])
	        elif a.find('unresolved staff comments') != -1:
	            trimmed_corpus.append(a[start_idx:a.find('unresolved staff comments') ])
	        elif a.find('properties') != -1:
	            trimmed_corpus.append(a[start_idx:a.find('properties') ])
	        
	    return trimmed_corpus


	# Create Risk Factor Corpus and Business Description Corpus
	years = list(corpus_dict.keys())

	corpus_business_description = {}
	corpus_risk_fctors = {}

	for i in years:
	    corpus_business_description.update({i : section_business(corpus_dict[i])})
	    corpus_risk_fctors.update({i : section_1a(corpus_dict[i])})
	    


	# Preprocessing

	for year in years:

	    # bus_desc
	    for i in range(len(corpus_business_description[year])):
	        bus_desc_list = text_preprocessing(corpus_business_description[year][i])

	        if bus_desc_list:
	            bus_desc_list = [year,'bus_desc'] + bus_desc_list

	            with open('./preprocess_output/' + year + '_' + str(i) + '_bus_desc.csv','w') as result_file:
	                wr = csv.writer(result_file, dialect='excel')
	                wr.writerow(bus_desc_list)

	    # risk_factor
	    for i in range(len(corpus_risk_fctors[year])):
	        risk_factor_list = text_preprocessing(corpus_risk_fctors[year][i])

	        if risk_factor_list:
	            risk_factor_list = [year,'risk_factor'] + risk_factor_list

	            with open('./preprocess_output/' + year + '_' + str(i) + '_risk_factor.csv','w') as result_file:
	                wr = csv.writer(result_file, dialect='excel')
	                wr.writerow(risk_factor_list)

    
if __name__ == "__main__":
    main()
