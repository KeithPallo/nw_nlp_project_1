#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

import nltk
from nltk.probability import FreqDist
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
nltk.download('wordnet')

from collections import Counter
from collections import OrderedDict

import string
import re
import unidecode
import requests
import json
from pprint import pprint

f = open('english.txt', 'r')
stop_words = f.read().splitlines()


pd.options.mode.chained_assignment = None  # default='warn'


# # Helper Functions ------------------------------------


def separate_awards(award_list):
    possible_people_awards = ['actor',' actress', 'musician', ' singer', 'composer', 'director', 'producer',
                        'screenwriter', 'stage technician', 'author']

    people_awards = []
    media_awards = []

    for category in award_list:
        if any(job in category.lower() for job in possible_people_awards):
            people_awards.append(category)
        else:
            media_awards.append(category)

    return people_awards, media_awards

def parse_award(award):
    """
    Returns a list of words that can be used to filter for a particular award
    """

    award = re.split('\W+', award)
    award = [i for i in award if i not in stop_words]
    award = list(set(award))
    return award

def get_awards_dict(awards_list):
    """
    Returns a dictionary that has all awards as keys, and a list of relevant filtering words as values
    """

    categories_dict = dict()
    for a in awards_list:
        terms = parse_award(a)
        categories_dict[a] = terms

    return categories_dict


def get_all_awards_tweets(award_list, categories_dict, data):
    """
    Using an award list and category dictionary, filters out tweets at an award level
    """

    d = {}
    for award in award_list:
        d["{0}".format(award)] = get_award_tweets(data, categories_dict[award])

    return d


def get_award_tweets(data, list1, spec = "people"):
    """
    Returns a list of tweets that are relevant to a particular award
    """
    synonyms = {}

    if spec == "people":
        synonyms = {
            'motion' : ['motion picture', 'motion', 'picture', 'movie'],
            'picture' : ['motion picture', 'motion', 'picture', 'movie'],
            'television' : ['television', 'tv'],
            'mini' : ['mini-series', 'mini', 'series', 'miniseries'],
            'series' : ['mini-series', 'mini', 'series', 'miniseries']
        }


    result = []

    for tweet in data:
        cond = True
        for i in list1:
            if i in synonyms:
                if all(j not in tweet.lower() for j in synonyms[i]):
                    cond = False
            elif i not in tweet.lower():
                cond = False
        if cond:
            result.append(tweet)

    return result


def compare_to_kb(nominees,kb):
    """
    Takes in a dictionary of potential nominees and removes those that don't appear in a relevant KB

    If no nominees are in the KB, then ... (currently top five)
    """

    final_nominees = {}

    for i in nominees:
        award_nominees = []

        for j in nominees[i]:
            if j[0].lower() in kb:
                award_nominees.append(j[0].lower())

        if not award_nominees:
            award_nominees = [""]

        award_nominees = list(set(award_nominees))
        final_nominees[i] = award_nominees

    return final_nominees


# # Award Categories Functions ----------------------------------


def clean_awards(text):
    " Cleans individual tweet for award search"

    remove_terms = ['#goldenglobes', 'golden globes', '#goldenglobe', 'golden globe', 'goldenglobes', 'goldenglobe', 'rt', 'golden', 'globe', 'globes']

    text = re.sub("(\s)#\w+","",text)    # strips away all hashtags
    text = re.sub("RT","",text)          # removes retweet
    text = re.sub("[^a-zA-Z ]", '',text) # removes all punctuation but keeps whitespace for tokenization
    text = text.lower()
    text = text.split()
    text = " ".join([term for term in text if term not in remove_terms]) #remove stop words

    return text


def find_tags(tweet):
    """
    Performs pos tagging at a tweet level
    """

    tokenizer = TweetTokenizer()
    tokens = tokenizer.tokenize(tweet)
    tags = nltk.pos_tag(tokens)
    return tags

def pos_search(tags,chunk_gram,label):

    potentials = ""
    chunk_parser = nltk.RegexpParser(chunk_gram)
    chunked = chunk_parser.parse(tags)
    for subtree in chunked.subtrees():
        if subtree.label() == label:
            raw_list = nltk.tag.util.untag(subtree)
            raw_list = [i for i in raw_list if wordnet.synsets(i)]
            string = ' '.join(raw_list)
            if "best" in string[0:6]:
                if len(string) >= len(potentials):
                    potentials = string

    if potentials == "":
        return "No Chunk"

    return potentials



def filter_df(df,label):

    data = df.loc[df[label] != "No Chunk"]
    data.drop(data.columns.difference([label]), 1, inplace=True)
    single_list = list(data[label])
    freq = FreqDist(single_list)

    return data, freq

def find_awards(df):
    """
    Returns a list of strings for all possible awards
    """
    # Shuffle data if necesarry
    sample_size = 200000
    if len(df['text']) > sample_size:
        df = df.sample(n=sample_size)

    # Clean awards, keep best, pos tag
    df['text'] = df['text'].apply(lambda x:  clean_awards(x))
    df_a = df[df['text'].str.contains("best")]
    df_a['tags'] = df_a['text'].apply(lambda x: find_tags(x))

    # Define regex patterns from generalized
    regex_pattern_0 = "P0: {<JJ.><NN.|JJ|VBG><...?>*<NN.>}"
    regex_pattern_1 = "P1: {<NN.><IN|NN.|IN><...?>*<NN.>}"
    regex_pattern_2 = "P2: {<RB.><JJ|NN.|VGB><...?>*<NN.|JJ>}"

    # Search for pos
    df_a['chunks_0'] = df_a['tags'].apply(lambda x: pos_search(x,regex_pattern_0,"P0"))
    df_a['chunks_1'] = df_a['tags'].apply(lambda x: pos_search(x,regex_pattern_1,"P1"))
    df_a['chunks_2'] = df_a['tags'].apply(lambda x: pos_search(x,regex_pattern_2,"P2"))

    data_0, freq_0 = filter_df(df_a,"chunks_0")
    data_1, freq_1 = filter_df(df_a,"chunks_1")
    data_2, freq_2 = filter_df(df_a,"chunks_2")

    freq = freq_0 + freq_1 + freq_2

    possible = []

    for i in freq.most_common():
        if i[1] >= 5: possible.append(i[0])

    return possible

# def find_awards(df):
#     """
#     Returns a list of strings for all possible awards
#     """
#     # Shuffle data if necesarry
#     sample_size = 200000
#     if len(df['text']) > sample_size:
#         df = df.sample(n=sample_size)
#
#     # Clean awards, keep best, pos tag
#     df_a = df[df['text'].str.contains('award')]
#     df_a['text'] = df_a['text'].apply(lambda x:  clean_awards(x))
#     # df_a = df[df['text'].str.contains("best")]
#     df_a['tags'] = df['text'].apply(lambda x: find_tags(x))
#
#     # Define regex patterns from generalized
#     regex_pattern_0 = "P0: {<JJ.><NN.|JJ|VBG><...?>*<NN.>}"
#     regex_pattern_1 = "P1: {<NN.><IN|NN.|IN><...?>*<NN.>}"
#     regex_pattern_2 = "P2: {<RB.><JJ|NN.|VGB><...?>*<NN.|JJ>}"
#
#     # Search for pos
#     df_a['chunks_0'] = df_a['tags'].apply(lambda x: pos_search(x,regex_pattern_0,"P0"))
#     df_a['chunks_1'] = df_a['tags'].apply(lambda x: pos_search(x,regex_pattern_1,"P1"))
#     df_a['chunks_2'] = df_a['tags'].apply(lambda x: pos_search(x,regex_pattern_2,"P2"))
#
#     data_0, freq_0 = filter_df(df_a,"chunks_0")
#     data_1, freq_1 = filter_df(df_a,"chunks_1")
#     data_2, freq_2 = filter_df(df_a,"chunks_2")
#
#     freq = freq_0 + freq_1 + freq_2
#
#     possible = []
#
#     for i in freq.most_common():
#         if i[1] >= 2: possible.append(i[0])
#
#     return possible


# # Hosts Function ---------------------------------------

# DATA PASSED IN AS LIST
def extract_hosts(data):
    # clean data
    cleaned_data = []
    
    for tweet in data:
        tt = TweetTokenizer(strip_handles=True, reduce_len=True, preserve_case=True)
        punctuation = list(string.punctuation)
        # strip stopwords, punctuation, url components 
        stop = stopwords.words('english') + punctuation + ['t.co', 'http', 'https', '...', '..', ':\\', 'RT', '#']
        strip_nums = re.sub("\d+", "", tweet)
        tokenized = tt.tokenize(strip_nums)
        terms_stop = [term for term in tokenized if term not in stop]
        cleaned = [term for term in terms_stop]
        cleaned = ' '.join(cleaned)
        cleaned_data.append(cleaned)
     
    
    # find host
    include_terms = ['host', 'hosted', 'hosting', 'hosts']
    remove_terms = ['next year']
    host = [];
    cohost = 0;
    for tweet in cleaned_data:
        if any(term in tweet for term in include_terms) and any(term not in tweet for term in remove_terms):
            host.append(tweet)
        if 'cohost' in tweet:
            cohost += 1
            
    bgrams = [];
    for tweet in host:
        bgrams += list(nltk.bigrams(tweet.split()))
        
    fdist = nltk.FreqDist(bgrams)
    
    if cohost > 5: #and len(final_hosts) > 1:
        fdist = fdist.most_common()
    else:
        fdist = fdist.most_common(1)
    
    final_hosts = []
    for host in fdist:
        name = host[0][0] + ' ' + host[0][1]
        final_hosts.append(name)
    
    return_list = []
    
    if cohost > 5:
        for name in final_hosts:
            if name in people_kb:
                return_list.append(name)
        return_list = return_list[:2]
    else:
        return_list.append(final_hosts[0])
    
    return return_list


# # Extract Functions -------------------------------------------

# ## Extract all instances of [people / media / presenters] from tweets for specific award

def extract_people(data, list1):
    """
    Extracts potential People nominees from an individual tweet
    """

    result = []

    translator = str.maketrans('', '', string.punctuation)
    remove_terms = ['#goldenglobes', 'golden globes', '#goldenglobe', 'golden globe', 'goldenglobes', 'goldenglobe', 'golden', 'globe', 'globes']
    stop = remove_terms + list1

    for tweet in data:

        tweet = re.sub("\d+", "", tweet)       #strip nums
        tweet = re.sub(r'http\S+', '', tweet)  #strip urls
        tweet = re.sub(r'#\S+', '', tweet)     #strip hashtags
        tweet = tweet.translate(translator)    #strip non-alphanumeric characters
        tweet = tweet.split()                  #tokenize
        tweet = [term for term in tweet if term.lower() not in stop_words] #remove stop words
        for i in stop:
            for j in tweet:
                if i.lower() in j.lower():
                    tweet.remove(j)
        result.append(tweet)



    grams = [];

    for tweet in result:
        if tweet:
            # Get all possible bigrams & trigrams in a tweet
            gram = list(nltk.everygrams(tweet, 2, 3))

            # Filter through and append to list for tweet
            for g in gram:
                if len(g) == 2:
                    if bool(re.match(r'\b[A-Z][a-z]+\b', g[0])) and bool(re.match(r'\b[A-Z][a-z]+\b', g[1])):
                        grams.append(' '.join(g))
                else:
                    if bool(re.match(r'\b[A-Z][a-z]+\b', g[0])) and bool(re.match(r'\b[A-Z][a-z]+\b', g[1])) and bool(re.match(r'\b[A-Z][a-z]+\b', g[2])):
                        grams.append(' '.join(g))


    fdist = nltk.FreqDist(grams)

    try:
        names = fdist.most_common()
    except:
        names = ""

    return names


def extract_media(data, list1):
    """
    Extracts potential media nominees from an individual tweet
    """

    result = []

    translator = str.maketrans('', '', string.punctuation)
    remove_terms = ['#goldenglobes', 'golden globes', '#goldenglobe', 'golden globe', 'goldenglobes', 'goldenglobe', 'golden', 'globe', 'globes', 'best']
    stop = remove_terms + list1

    for tweet in data:
        tweet = re.sub("\d+", "", tweet)      #strip nums
        tweet = re.sub(r'http\S+', '', tweet) #strip urls
        tweet = re.sub(r'#\S+', '', tweet)    #strip hashtags
        tweet = tweet.translate(translator)   #strip non-alphanumeric characters
        tweet = tweet.split()                 #tokenize
        for i in stop:
            for j in tweet:
                if i.lower() in j.lower():
                    tweet.remove(j)
        tweet = ' '.join(tweet)
        result.append(tweet)


    grams = [];

    for tweet in result:
        if tweet:

            grams.extend(re.findall(r"([A-Z][\w-]*(?:\s+[A-Z][\w-]*)+)", tweet))
            grams.extend(re.findall(r"\b[A-Z][a-z]+\b.*\b[A-Z][a-z]+\b", tweet))
            #singular = re.findall(r"\b[A-Z][a-z]+\b", tweet)
            #singular = [i for i in singular if not wordnet.synsets(i)]
            #grams.extend(singular)

    fdist = nltk.FreqDist(grams)

    try:
        names = fdist.most_common()

    except:
        names = ""

    return names


def extract_presenters(data, list1, winners):
    result = []

    translator = str.maketrans('', '', string.punctuation)
    remove_terms = ['#goldenglobes', 'golden globes', '#goldenglobe', 'golden globe', 'goldenglobes', 'goldenglobe', 'golden', 'globe', 'globes']

    if winners:
        stop = remove_terms + list1 + winners.split()
    else:
        stop = remove_terms + list1

    for tweet in data:

        tweet = re.sub("\d+", "", tweet) #strip nums
        tweet = re.sub(r'http\S+', '', tweet) #strip urls
        tweet = re.sub(r'#\S+', '', tweet) #strip hashtags
        tweet = tweet.translate(translator) #strip non-alphanumeric characters
        tweet = tweet.split() #tokenize

        for i in stop:
            for j in tweet:
                if i.lower() in j.lower():
                    tweet.remove(j)
        result.append(tweet)


    grams = [];

    for tweet in result:
        if tweet:
            gram = list(nltk.everygrams(tweet, 2, 3))
            for g in gram:
                if len(g) == 2:
                    if bool(re.match(r'\b[A-Z][a-z]+\b', g[0])) and bool(re.match(r'\b[A-Z][a-z]+\b', g[1])):
                        grams.append(' '.join(g))
                else:
                    if bool(re.match(r'\b[A-Z][a-z]+\b', g[0])) and bool(re.match(r'\b[A-Z][a-z]+\b', g[1])) and bool(re.match(r'\b[A-Z][a-z]+\b', g[2])):
                        grams.append(' '.join(g))


    fdist = nltk.FreqDist(grams)


    try:
        names = fdist.most_common()
    except:
        names = ""

    return names


# # Get Functions

def get_nominees(award_list, categories_dict, tweets_dict, spec = ""):
    """
    Gets all potential nominees based on extract_media or extract_people
    """

    if spec == "people":
        funct = extract_people
    elif spec == "media":
        funct = extract_media
    else:
        print("there is a problem")

    nominees = {}
    for award in award_list:
        nominees["{0}".format(award)] = funct(tweets_dict[award], categories_dict[award])

    return nominees


def get_presenters(award_list, categories_dict, tweets_dict, winners, people_kb):
    # Initialize return dictionary
    present = ['present', 'annouc', 'introduc']

    # Remove tweets that do not have presenters keyswords in them (currently using stemming)
    for award in award_list:
        for tweet in tweets_dict[award]:
            if all(i not in tweet for i in present):
                tweets_dict[award].remove(tweet)

    # Initialize return dictionary
    presenters = {}

    for award in award_list:
        all_presenters = extract_presenters(tweets_dict[award], categories_dict[award], winners[award])

        # Check to see if there are potential matches (all_presenters is a list of tuples)
        if all_presenters != "nothing_here":

            found = False

            # select the most common potential presenter that is a person (from pinging people_kb)

            count = 0
            names = []

            for potential in all_presenters:
                if count > potential[1]:
                    break
                if potential[0].lower() in people_kb:
                    names.append(potential[0].lower())
                    count = potential[1]
                    found = True

            presenters[award] = names

            # account for no valid presenters from frequency distibution
            if not found: presenters[award] = []

        # account for empty frequency distribution
        else:
            presenters[award] = []

    return presenters

def get_media_winners(nominees):
    final_winners = {}

    for award in nominees:
        winner = ''.join(nominees[award][0])
        final_winners[award] = winner

    return final_winners


def get_people_winners(nominees):
    " Gets the most frequent nominees from passed in dictionary"
    final_winners = {}

    for award in nominees:
        #print(award)
        winner = ''.join(nominees[award][0][0])
        final_winners[award] = winner.lower()

    return final_winners



def get_dressed(data, kb):
    
    result = []
       
    translator = str.maketrans('', '', string.punctuation)
    remove_terms = ['#goldenglobes', 'golden globes', '#goldenglobe', 'golden globe', 'goldenglobes', 'goldenglobe', 'golden', 'globe', 'globes']
    stop = remove_terms
    include_terms = ['dress', 'fashion', 'red', 'carpet', 'haute couture', 'gown', 'design', 'look']
    
    for tweet in data:
        
        tweet = re.sub("\d+", "", tweet)       #strip nums
        tweet = re.sub(r'http\S+', '', tweet)  #strip urls
        tweet = re.sub(r'#\S+', '', tweet)     #strip hashtags
        tweet = tweet.translate(translator)    #strip non-alphanumeric characters

        for i in stop:
            for j in tweet:
                if i.lower() in j.lower():
                    tweet.remove(j)
        
        if any(term in tweet for term in include_terms):
            result.append(tweet)
    
    
    # extract people and put in dictionary with compound scores
    sentiment_analyzer = SentimentIntensityAnalyzer()
    score_dict = {}
    best_dressed_dict = {}
    worst_dressed_dict = {}

    for tweet in result:
        all_scores = sentiment_analyzer.polarity_scores(tweet)
        for k in sorted(all_scores):
            if k == 'compound':
                useful_score = all_scores[k]

        if tweet:
            # Get all possible bigrams & trigrams in a tweet
            gram = list(nltk.everygrams(tweet.split(), 2, 3))

            # Filter through and append to list for tweet
            for g in gram:
                if len(g) == 2:
                    if bool(re.match(r'\b[A-Z][a-z]+\b', g[0])) and bool(re.match(r'\b[A-Z][a-z]+\b', g[1])):
                        name = ' '.join(g).lower()
                        if useful_score > 0:
                            if name in best_dressed_dict:
                                best_dressed_dict[name] += useful_score
                            else:
                                best_dressed_dict[name] = useful_score
                        if useful_score < 0:
                            if name in worst_dressed_dict:
                                worst_dressed_dict[name] += useful_score
                            else:
                                worst_dressed_dict[name] = useful_score
                else:
                    if bool(re.match(r'\b[A-Z][a-z]+\b', g[0])) and bool(re.match(r'\b[A-Z][a-z]+\b', g[1])) and bool(re.match(r'\b[A-Z][a-z]+\b', g[2])):
                        name = ' '.join(g).lower()
                        if useful_score > 0:
                            if name in best_dressed_dict:
                                best_dressed_dict[name] += useful_score
                            else:
                                best_dressed_dict[name] = useful_score
                        if useful_score < 0:
                            if name in worst_dressed_dict:
                                worst_dressed_dict[name] += useful_score
                            else:
                                worst_dressed_dict[name] = useful_score
        
        
    # look in kb for matches     
    final_dict = {}
    for best, worst in zip(best_dressed_dict, worst_dressed_dict):
        if best.lower() in kb:
            final_dict[best] = best_dressed_dict[best]
        if worst.lower() in kb:
            final_dict[worst] = worst_dressed_dict[worst]
            

    # get key with max value
    best_dressed = []
    worst_dressed = []
    
    while len(best_dressed) < 5:
        best_person = max(final_dict.items(), key=lambda k: k[1])[0]
        worst_person = min(final_dict.items(), key=lambda k: k[1])[0]
        best_dressed.append(best_person)
        worst_dressed.append(worst_person)
        del final_dict[best_person]
        del final_dict[worst_person]
    
    return best_dressed, worst_dressed


# # Wrapper Functions ------------------------------------


def compress_associated_dict(award_list,nominees,winners,presenters):

    our_dict = {}

    for award in award_list:
        our_dict[award] = {
            'nominees' : nominees[award],
            'winner' : winners[award],
            'presenters' : presenters[award]
        }

    return our_dict

def associated_tasks(award_list,data,data_presenter,spec,kb,kb2):

    # Create a dictionary to filter tweets at a category level
    cat_filter_dict = get_awards_dict(award_list)

    # Get all associated tweets for each award
    tweets_dict = get_all_awards_tweets(award_list, cat_filter_dict, data)


    # For each award, get all associated nominees
    full_nom_dict = get_nominees(award_list, cat_filter_dict, tweets_dict, spec)

    # Filter out all nominees that are not in the dictionary
    final_nom = compare_to_kb(full_nom_dict, kb)

    # Extract final winners - different for people and media
    final_winners = {}
    if spec == "media":
        final_winners = get_media_winners(final_nom)

    elif spec == "people":
        final_winners = get_people_winners(full_nom_dict)

    # Get possible presenters
    tweets_pres_dict = get_all_awards_tweets(award_list, cat_filter_dict, data_presenter)
    final_pres = get_presenters(award_list, cat_filter_dict, tweets_pres_dict, final_winners,kb2)

    return final_nom, final_winners,final_pres


def human_readable(award_list, hosts, final_nom, final_winner, final_pres, best_dressed, worst_dressed):
    
    f = open("human_readable_results.txt", "w")
    f.write('Hosts: ' + ', '.join(hosts) + '\n\n')
    
    for award in award_list:
        f.write('Award: ' + award + '\n')
        f.write('Presenters: ' + ', '.join(final_pres[award]) + '\n')
        f.write('Nominees: ' + ', '.join(final_nom[award]) + '\n')
        f.write('Winner: ' + final_winner[award] + '\n\n')
        
    f.write('Best Dressed: ' + ', '.join(best_dressed) + '\n')
    f.write('Worst Dressed: ' + ', '.join(worst_dressed) + '\n')
    
    return


def main_exec(award_list,df,kb_p,kb_m):
    """
    Main execution file - how you run the program
    Itype: kb_p and kb_m are sets for our built KB's
    """

    # Presenter dataframe
    df_presenter = df[df['text'].str.contains('present')]

    sample_size = 400000
    if len(df['text']) > sample_size:
        df = df.sample(n=sample_size)

    data = df['text'].values.tolist()
    data_presenter = df_presenter['text'].values.tolist()



    # Segment out awards award categories
    people_awards, media_awards = separate_awards(award_list)


    # Call search function - winner, nominee, presenter

    final_nom, final_winner,final_pres = associated_tasks(people_awards, data, data_presenter, "people", kb_p, kb_p)

    media_nom, media_winner, media_pres = associated_tasks(media_awards, data, data_presenter,"media", kb_m, kb_p)

    final_nom.update(media_nom)
    final_winner.update(media_winner)
    final_pres.update(media_pres)


    # Call host search function
    hosts = extract_hosts(data)

    # Call award recognition function
    awards = find_awards(df)
    
    best_dressed, worst_dressed = get_dressed(data, kb_p)
    human_readable(award_list, hosts, final_nom, final_winner, final_pres, best_dressed, worst_dressed)

    return hosts, awards, final_nom, final_winner, final_pres
