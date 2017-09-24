import pandas as pd

# df = pd.read_csv('cities1000.txt', sep='\t', header=19)
# df.to_csv('cities1000.csv', index=False)

df = pd.read_csv('worldcitiespop.txt', sep=',', encoding="ISO-8859-1", low_memory=False)
cities = df.set_index('City')['Country'].to_dict()
#print(cities)

df = pd.read_csv('CountryAbbreviations.txt', sep='\t')
#print(df)
abbreviations = df.set_index('A2 (ISO)')['COUNTRY'].to_dict()
#print(abbreviations)


def location(sentence):
    words = sentence.split();
    ret = []
    for word in words:
        if word.lower() in cities.keys():
            ret.append(word + ', ' + abbreviations[cities[word.lower()].upper()])
    if len(ret) == 0:
        return None
    return ret

sentence1 = "Durham"
print(cities['durham'])
print(location(sentence1))
#print(cities.keys())
