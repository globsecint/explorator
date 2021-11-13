from collections import Counter
import numpy as np
import pandas as pd 
import twint
import spacy
import geocoder
import warnings
warnings.filterwarnings('ignore')

print('GSI Named Entity Recognizer ver 0.1 alpha')
print('-----Starting NER process-----')

# Configure Spacy NLP
nlp = spacy.load("en_core_web_trf")

## Configure search parameters
c = twint.Config()
c.Limit = 100000
c.Store_csv = True
output_names = []
ignore= ['']
special_country = ['']
processed = list()
associations = list()

def tocsv(df, filename):
    df.to_csv(filename)
    return 

print('Setting parameters: done')

test_data = pd.read_csv('news.csv')

print('Data loaded')

notes =  test_data['Titles']
size = len(notes)

time = str(size/1000)

print('Running NLP model')

print('Load time: '+time+' minutes')

n = size
if n > 2000:
    t = 1
    divided = [notes[i:i+1000] for i in range(0, len(notes), 1000)]
    for divide in divided:
        print('loading...')
        for note in divide:
             result = nlp(note)
             processed.append((set([(ent.text,ent.label_) for ent in result.ents])))
        print('Time remaining: '+str(float(time)-t)+' minutes')
        t = t+1

else: 
    print('loading...')
    for note in notes[:int(size-(size-n))]:
        result = nlp(note)
        processed.append((set([(ent.text,ent.label_) for ent in result.ents])))

print('NLP model: Done')
print('Creating datasets')
all = [val for sublist in processed for val in sublist]
all_ents = list(set([val for sublist in processed for val in sublist]))
ent_dict = dict(all_ents)

names = [k[0] for k in all_ents]
tags = [k[1] for k in all_ents]

associations = []
for name in names: 
    for entry in processed:
        entry_names = [k[0] for k in entry]
        if name in entry_names:
            for i in entry:
                associations.append((all_ents[names.index(name)][0],all_ents[names.index(name)][1],i[0],i[1]))

asoc_list = pd.DataFrame(associations,columns=['Entity1','Type 2','Entity2','Type 2'])

def tocsv(df, filename):
    df.to_csv(filename)
    return 

tocsv(asoc_list,'associations.csv')

print('Created: associations.csv')


observations_db3= pd.DataFrame(columns=['Entity','Type','Title','Time','Entity 2','Type 2'])
observations_db3['Entity']=list(range(size))
ent_dict = dict(all_ents)
sources = test_data['Source']

for name in names:
    for i in list(range(size)): 
        if name in notes[i]:
            title = notes[i]
            for a in processed[i]:
                observations_db3.loc[i] =pd.Series({'Entity':name,'Type':ent_dict[name],'Time': None,'Title':notes[i],'Entity 2':a[0],'Type2':a[1]})


tocsv(observations_db3,'observations3.csv')
print('Created: observations.csv')
print('-----NER process ended-----')