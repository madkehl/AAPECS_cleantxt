# script finds the median cos(theta) between vectors of entries
# gives a sense of average similarity between consecutive entries


# command line loads
# python -m spacy download en_core_web_md; md has word vector options

import pandas as pd
import numpy as np
import statistics as ST
from numpy import linalg as LA
import random
import spacy


nlp = spacy.load("en_core_web_md")
total_df = pd.read_csv('/Users/madke/Documents/AAPECSMadeline.csv', index_col = 0)
# text to spacy objects


def convert_to_spacy(series):
    """
    series:  a column from df containing sentences pretty much
    """
    vector_ls = []
    nlp_ls = []
    for i in series:
        i = str(i)
        if len(i) > 1:
            nlp_un  = nlp(i)
            nlp_ls.append(nlp_un)
            vector_ls.append(nlp_un.vector)
        else:
            nlp_ls.append(None)
            vector_ls.append(None)
    return nlp_ls, vector_ls


# calculating values for cos(theta); not theta; these values transformed in r
def get_thetas(vec_ls, randomize=False):
    result_ls = []
    clean_vec = vec_ls.dropna()
    print(vec_ls)
    if randomize is True:
        if len(vec_ls) > 1:
            vec_ls_rand = random.sample(list(vec_ls), k=len(vec_ls))
            clean_vec_ls = [np.dot(t, s) / (LA.norm(t) * LA.norm(s)) if (s is not None and t is not None) else None for s, t in zip(vec_ls_rand, vec_ls_rand[1:])]
        else:
            clean_vec_ls = None
    else:
        clean_vec_ls = [np.dot(t, s) / (LA.norm(t) * LA.norm(s)) for s, t in zip(clean_vec, clean_vec[1:])]
    if len(clean_vec_ls) > 0:
        result_ls.append(ST.median(clean_vec_ls))
    else:
        result_ls.append(None)
    return result_ls


nlp_list_s, nlp_swv = convert_to_spacy(total_df['stressDescription'])
nlp_list_u, nlp_uwv = convert_to_spacy(total_df['unpleasantEventDescription'])
nlp_list_p, nlp_pwv = convert_to_spacy(total_df['pleasantEventDescription'])


# grouped data frame to allow for aggregating based on pID

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


for i in indices:
    temp = nlp_df_grouped.get_group(i)
    swv_cons = get_thetas(temp['nlp_swv'])
    swv_rand = get_thetas(temp['nlp_swv'], randomize=True)
    uwv_cons = get_thetas(temp['nlp_uwv'])
    uwv_rand = get_thetas(temp['nlp_uwv'], randomize=True)
    pwv_cons = get_thetas(temp['nlp_pwv'])
    pwv_rand = get_thetas(temp['nlp_pwv'], randomize=True)


nlp_agg_df = pd.DataFrame({
    
    'participant_id': indices,
    'swvDmed': swv_cons,
    'uwvDmed': uwv_cons,
    'pwvDmed':pwv_cons,
    'swvVar': swv_rand,
    'uwvVar': uwv_rand,
    'pwvVar':pwv_rand
})


nlp_agg_df.to_csv('./aapecs_rigidity_estimates.csv')


