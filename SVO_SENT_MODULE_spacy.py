#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 22:28:48 2017

@author: ruobingwang
"""
from spacy.en import English
import file_io
import spacy
import pandas as pd
from nltk import data
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import datefinder
import re
import datetime
from dateparser import parse
import pandas as pd
from pycorenlp import StanfordCoreNLP

class SVOSENT(object):
    """
    Class Methods to Extract Subject Verb Object Tuples and sentiments from a Sentence
    """

    def __init__(self, language='english'):
        """
        Initialize 
        """
        self.nlp = spacy.load('en')  # spacy parser
        self.sent_detector = data.load('tokenizers/punkt/english.pickle')
        self.analyzer = SentimentIntensityAnalyzer()  # for sentiment analysis

    def getTexts(self, directory):
        # function by Tye
        # Input: Directory
        # Output:List of all text files in the directory fully loaded into memory
        texts = []
        pathnames = file_io.getFilesRecurse(directory, '.txt')
        for pathname in pathnames:
            texts.append(file_io.openFile(pathname))
        return texts

    def split_and_clean(self, text):
        '''
        Temporay function only useful for corpus data
        '''
        textlist = text.split('______________________________________________________')
        result = [text[text.find("Full text:") + 10:text.find("Publication title")] for text in textlist if
                  len(text) != 0]
        return result

    def get_svo(self, sentence):
        '''
        get SVO of single sentence
        '''
        parsed_phrase = self.nlp(sentence)
        names = list(parsed_phrase.ents)
        corrected_names = []
        for e in names:
            linked = e.text
            if any(map(str.isupper, linked)) and any(
                            ext in linked for ext in ['January', 'February', 'March', 'April', 'May',
                                                      'June', 'July', 'August', 'September', 'October', 'November',
                                                      'December', 'Sunday', 'Monday',
                                                      'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'week',
                                                      'Yesterday', 'month', 'day', 'Today']) == False:
                corrected_names.append(linked)
        subjects = []
        objects = []
        verbs = []
        for text in parsed_phrase:
            if text.dep_.startswith("nsubj") or text.dep_ in ['conj']:
                subject = text.orth_
                subjects.append(subject)
            if text.dep_ in ["dobj", 'pobj']:
                object_ = text.orth_
                objects.append(object_)
            if text.pos_ == "VERB":
                verb = text.orth_
                verbs.append(verb)

        try:
            event_date = list(set(sentence.replace('.', '').split()) & set(['Monday', 'Tuesday', 'Wednesday', 'Tursday',
                                                                            'Friday', 'Saturday', 'Sunday', 'Today',
                                                                            'today',
                                                                            'Tomorrow', 'tomorrow', 'Yesterday',
                                                                            'yesterday']))[0]

        except:
            try:
                event_date = list(datefinder.find_dates(sentence))[0]
                if str(event_date.year) not in sentence:
                    event_date = str(event_date.month) + '/' + str(event_date.day)
                event_date = str(event_date)
            except:
                event_date = None

        return {'Sentence': sentence,
                'Subjects': subjects,
                'Predicates': verbs,
                'Objects': objects,
                'Names': corrected_names,
                'Event_date': event_date}

    def get_svo_from_article(self, article):
        sentences = self.sentence_split(article)
        val = []
        for sent in sentences:
            svoresult = self.get_svo(sent)
            val.append(svoresult)
        return pd.DataFrame(val)

    def sentence_split(self, text):
        sentences = self.sent_detector.tokenize(text)
        return sentences

    def sentimentAnalysis(self, sentence):
        result = self.analyzer.polarity_scores(sentence)
        result['Sentence'] = sentence
        return result

    def get_senti_from_article(self, article):
        sentences = self.sentence_split(article)
        val = []
        for sent in sentences:
            result = self.sentimentAnalysis(sent)
            val.append(result)
        return pd.DataFrame(val)

    ###############################################
    # get both SVO and sent in one dataframe


    def svo_senti_from_article(self, article, subject=None):
        try:
            date = list(datefinder.find_dates(article))[-1]
        except:
            date = None
        sentences = self.sentence_split(article)
        val1 = []
        val2 = []

        for sent in sentences:
            val1.append(self.sentimentAnalysis(sent))
            val2.append(self.get_svo(sent))
        result = pd.merge(pd.DataFrame(val1), pd.DataFrame(val2), on='Sentence')[
            ['Sentence', 'Names', 'Subjects', 'Predicates', 'Objects', 'compound', 'pos', 'neu', 'neg', 'Event_date']]
        #        try:
        #            result['date']=date
        #        except:
        #            result['date']='-----'
        result['Article_date'] = date

        def correctdate(eventdate, articledate):
            if eventdate == None:
                return None
            if articledate == None:
                return None
            try:
                corrected_date = parse(eventdate, settings={'RELATIVE_BASE': articledate})
            except:
                corrected_date = None
            return corrected_date

        result['Event_date'] = result['Event_date'].apply(lambda x: correctdate(x, date))
        #        try:
        #            result.loc[result['date']> datetime.datetime.today() + datetime.timedelta(days=1),'date']='-----'
        #        except:
        #            pass
        result = result.drop_duplicates(subset=['Sentence'], keep='first')  # remove duplicate rows
        if subject == None:
            return result
        else:
            return result[result['Names'].apply(lambda x: subject in x)]

    def WriteCSV(self, df, name):
        df.to_csv(name + '.csv', index=False)


if __name__ == "__main__":
    nlp = StanfordCoreNLP('http://localhost:9000')

    print('Importing cities...')
    df = pd.read_csv('worldcitiespop.txt', sep=',', encoding="ISO-8859-1", low_memory=False)
    cities = df.set_index('City')['Country'].to_dict()
    print('Done')

    print('Importing abbreviations...')
    df = pd.read_csv('CountryAbbreviations.txt', sep='\t')
    # print(df)
    abbreviations = df.set_index('A2 (ISO)')['COUNTRY'].to_dict()
    print('Done')


    def locations(sentence):
        output = nlp.annotate(sentence, properties={
            'annotators': 'ner',
            'outputFormat': 'json'
        })
        location_finished = True
        location_list = []
        location_phrase = ""
        #print(output)
        for token in output['sentences'][0]['tokens']:
            if token['ner'] == 'LOCATION':
                location_phrase += token['word'] + " "
                location_finished = False
            else:
                location_finished = True
            if location_finished:
                location_phrase = location_phrase.strip()
                if location_phrase.lower() in cities.keys():  # checks if it's a city
                    location_list.append(
                        location_phrase)
                location_phrase = ""

        return location_list


    sentence1 = 'United States and China met up in Durham.'
    print(locations(sentence1))
    print("Ruobing's code")
    svo_sent = SVOSENT()

    article = 'United States and China met in Durham.'
    result = svo_sent.svo_senti_from_article(article)
    print(result)
    print(type(result))
    print(result['Objects'])
    print(result['Objects'][0])
    '''
    articles_not=svo_sent.getTexts('corpus4')[-1]
    articles=svo_sent.split_and_clean(articles_not)
    import time
    t0=time.time()
    results=[]
    for i,article in enumerate(articles):
        try:
            result=svo_sent.svo_senti_from_article(article)
            results.append(result)
            print(i,end='th/')
            print(len(articles),end='')
            print(' article is done')
        except:
            print(i,' th article is empty!')
    #result2=svo_sent.svo_senti_from_article(article,'Robin')
    t1=time.time()
    results=pd.concat(results, axis=0)
    print('time cost',end=':')
    print(t1-t0)
    #print(results)
    svo_sent.WriteCSV(results,'corpus4_full_dataset')
    '''
