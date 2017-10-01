import spacy
import pandas as pd

nlp = spacy.load('en')

print('Importing countries...')
df = pd.read_csv('CountryAbbreviations.txt', sep='\t')
abbreviations = df.set_index('COUNTRY')['A2 (ISO)'].to_dict()
print('Done')


def city_parser(sentence):
    doc = nlp(sentence)
    location_list = []
    #for ent in doc.ents:  # used for debugging - sometimes the entity recognition is incorrect
        #print(ent.label_, ent.text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            if ent.text not in abbreviations.keys():
                location_list.append(ent.text)
    return location_list


print(abbreviations.keys())
sentence1 = 'China and Raleigh had a lot of meetings.'
print(city_parser(sentence1))
