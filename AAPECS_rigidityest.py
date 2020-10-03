### script finds the median cos(theta) between vectors of entries
### gives a sense of average similarity between consecutive entries

import pandas as pd
import numpy as np
import statistics as ST
from numpy import linalg as LA

import nltk
from nltk import pos_tag
import spacy

#command line loads
#python -m spacy download en_core_web_md; md has word vector options

nlp = spacy.load("en_core_web_md")

total_df = pd.read_csv('/Users/madke/Documents/AAPECSMadeline.csv', index_col = 0)

#text to spacy objects
nlp_list_s = []
for i in total_df['stressDescription']:
    i = str(i)
    if len(i) > 1:
        nlp_list_s.append(nlp(i))
    else:
        nlp_list_s.append(None)
        
nlp_list_u = []
for i in total_df['unpleasantEventDescription']:
    i = str(i)
    if len(i) > 1:
        nlp_list_u.append(nlp(i))
    else:
        nlp_list_u.append(None)
        
nlp_list_p = []
for i in total_df['pleasantEventDescription']:
    i = str(i)
    if len(i) > 1:
        nlp_list_p.append(nlp(i))
    else:
        nlp_list_p.append(None)

#nlp objects to word vectors
nlp_swv = []

for i in nlp_list_s:
    if i == None:
        nlp_swv.append(None)
    else:
        nlp_swv.append(i.vector)

nlp_uwv = []

for i in nlp_list_u:
    if i == None:
        nlp_uwv.append(None)
    else:
        nlp_uwv.append(i.vector)
               
nlp_pwv = []

for i in nlp_list_p:
    if i == None:
        nlp_pwv.append(None)
    else:
        nlp_pwv.append(i.vector)

#grouped data frame to allow for aggregating based on pID

nlp_df = pd.DataFrame({
    "participant_id": total_df['participantID'],
    "nlp_s": nlp_list_s,
    "nlp_u":nlp_list_u,
    "nlp_p":nlp_list_p,
    "nlp_swv": nlp_swv,
    "nlp_uwv":nlp_uwv,
    "nlp_pwv": nlp_pwv
    
})

nlp_df_grouped = nlp_df.groupby('participant_id')
indices = list(set(total_df['participantID']))

swvDagg= []
uwvDagg = []
pwvDagg = []

#calculating values for cos(theta); not theta; these values transformed in r 
for i in indices:
    temp = nlp_df_grouped.get_group(i)
    lwds = temp['nlp_swv'].dropna()
    swvD = [np.dot(t,s)/(LA.norm(t)*LA.norm(s)) for s, t in zip(lwds, lwds[1:])]
    lwdu = temp['nlp_uwv'].dropna()
    uwvD = [np.dot(t,s)/(LA.norm(t)*LA.norm(s))for s, t in zip(lwdu, lwdu[1:])]
    lwdp = temp['nlp_pwv'].dropna()
    pwvD = [np.dot(t,s)/(LA.norm(t)*LA.norm(s)) for s, t in zip(lwdp, lwdp[1:])]
    if len(swvD) > 0:
        swvDagg.append(ST.median(swvD))
    else:
        swvDagg.append(None)
    if len(uwvD) > 0:
        uwvDagg.append(ST.median(uwvD))
    else:
        uwvDagg.append(None)
    if len(pwvD) > 0:
        pwvDagg.append(ST.median(pwvD))
    else:
        pwvDagg.append(None)
        

swvVar = []
uwvVar = []
pwvVar = []

for i in indices:
    temp = nlp_df_grouped.get_group(i)
    lwds = temp['nlp_swv'].dropna()
    lwds = [i for i in lwds if len(i) > 1]
    if len(lwds) > 1:
        lwds1 = random.sample(list(lwds), k = len(lwds)) +  random.sample(list(lwds),  k = len(lwds)) +  random.sample(list(lwds),  k = len(lwds))
        swvD = [np.dot(t,s)/(LA.norm(t)*LA.norm(s)) for s, t in zip(lwds1, lwds1[1:])]
        swvVar.append(np.median(swvD))
    else:
        swvVar.append(None)
    lwdu = temp['nlp_uwv'].dropna()
    if len(lwdu) > 1:
        lwdu1 = random.sample(list(lwdu), k = len(lwdu)) +  random.sample(list(lwdu), k = len(lwdu)) +  random.sample(list(lwdu), k = len(lwdu))
        uwvD = [np.dot(t,s)/(LA.norm(t)*LA.norm(s)) for s, t in zip(lwdu1, lwdu1[1:])]
        uwvVar.append(np.median(uwvD))
    else:
        uwvVar.append(None)
    lwdp = temp['nlp_pwv'].dropna()
    if len(lwdp) > 1:
        lwdp1 = random.sample(list(lwdp), k = len(lwdp)) +  random.sample(list(lwdp), k = len(lwdp)) +  random.sample(list(lwdp), k = len(lwdp))
        pwvD = [np.dot(t,s)/(LA.norm(t)*LA.norm(s)) for s, t in zip(lwdp1, lwdp1[1:])]
        pwvVar.append(np.median(pwvD))
    else:
        pwvVar.append(None)
   

aggnlp_df = pd.DataFrame({
    
    'participant_id': indices,
    'swvDmed': swvDagg,
    'uwvDmed': uwvDagg,
    'pwvDmed':pwvDagg
    'swvVar': swvVar,
    'uwvVar': uwvVar,
    'pwvVar':pwvVar
})


aggnlp_df.to_csv('/users/madke/documents/012320AAPECS_rigidity.csv')


