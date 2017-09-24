import pandas as pd
from pycorenlp import StanfordCoreNLP

# Make sure your server is connected
# Daniel\Downloads\stanford etc.
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port 9000 -timeout 15000
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


# print(output['sentences'][0]['tokens'])
# print


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


sentence1 = 'Durham, Raleigh, Edison, Mountainview, and Palo Alto all have great coffee.'
print(locations(sentence1))
