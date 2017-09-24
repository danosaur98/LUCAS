import spacy
import pandas as pd

nlp = spacy.load('en')

print('Importing cities...')
df = pd.read_csv('worldcitiespop.txt', sep=',', encoding="ISO-8859-1", low_memory=False)
cities = df.set_index('City')['Country'].to_dict()
print('Done')


def city_parser(sentence):
    doc = nlp(sentence)
    location_list = []
    # for ent in doc.ents: #used for debugging - sometimes the entity recognition is incorrect
    #    print(ent.label_, ent.text)
    for ent in doc.ents:
        if ent.label_ == "GPE":
            if ent.text.lower() in cities.keys():
                location_list.append(ent.text)
    return location_list


s1 = "Raleigh, Durham, Trenton and Palo Alto are all in the United States of America"
print(city_parser(s1))
