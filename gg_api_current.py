#!/usr/bin/env python
# coding: utf-8

# In[ ]:

# Command line functions

# Pre-ceremony call
# python -c 'import PleaseWork; PleaseWork.pre_ceremony()'

# Main function call
# python -c 'import gg_api_current; gg_api_current.main(option=False)'

# Autograder call
# python autograder.py

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

from wrapped_3 import *

OFFICIAL_AWARDS_1315 = ['cecil b. demille award', 'best motion picture - drama', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best motion picture - comedy or musical', 'best performance by an actress in a motion picture - comedy or musical', 'best performance by an actor in a motion picture - comedy or musical', 'best animated feature film', 'best foreign language film', 'best performance by an actress in a supporting role in a motion picture', 'best performance by an actor in a supporting role in a motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best television series - comedy or musical', 'best performance by an actress in a television series - comedy or musical', 'best performance by an actor in a television series - comedy or musical', 'best mini-series or motion picture made for television', 'best performance by an actress in a mini-series or motion picture made for television', 'best performance by an actor in a mini-series or motion picture made for television', 'best performance by an actress in a supporting role in a series, mini-series or motion picture made for television', 'best performance by an actor in a supporting role in a series, mini-series or motion picture made for television']
OFFICIAL_AWARDS_1819 = ['best motion picture - drama', 'best motion picture - musical or comedy', 'best performance by an actress in a motion picture - drama', 'best performance by an actor in a motion picture - drama', 'best performance by an actress in a motion picture - musical or comedy', 'best performance by an actor in a motion picture - musical or comedy', 'best performance by an actress in a supporting role in any motion picture', 'best performance by an actor in a supporting role in any motion picture', 'best director - motion picture', 'best screenplay - motion picture', 'best motion picture - animated', 'best motion picture - foreign language', 'best original score - motion picture', 'best original song - motion picture', 'best television series - drama', 'best television series - musical or comedy', 'best television limited series or motion picture made for television', 'best performance by an actress in a limited series or a motion picture made for television', 'best performance by an actor in a limited series or a motion picture made for television', 'best performance by an actress in a television series - drama', 'best performance by an actor in a television series - drama', 'best performance by an actress in a television series - musical or comedy', 'best performance by an actor in a television series - musical or comedy', 'best performance by an actress in a supporting role in a series, limited series or motion picture made for television', 'best performance by an actor in a supporting role in a series, limited series or motion picture made for television', 'cecil b. demille award']

def get_hosts(year):
    '''Hosts is a list of one or more strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    y = year[2:]

    with open('%s_hosts.json' % y, 'r') as f:
        data = json.load(f)

    hosts = json.loads(data)

    return hosts

def get_awards(year):
    '''Awards is a list of strings. Do NOT change the name
    of this function or what it returns.'''
    # Your code here
    y = year[2:]

    with open('%s_awards.json' % y, 'r') as f:
        data = json.load(f)

    awards = json.loads(data)

    return awards

def get_nominees(year):
    '''Nominees is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change
    the name of this function or what it returns.'''
    # Your code here
    y = year[2:]

    with open('%s_nominees.json' % y, 'r') as f:
        data = json.load(f)

    nominees = json.loads(data)

    return nominees

def get_winner(year):
    '''Winners is a dictionary with the hard coded award
    names as keys, and each entry containing a single string.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    y = year[2:]

    with open('%s_winners.json' % y, 'r') as f:
        data = json.load(f)

    winners = json.loads(data)

    return winners

def get_presenters(year):
    '''Presenters is a dictionary with the hard coded award
    names as keys, and each entry a list of strings. Do NOT change the
    name of this function or what it returns.'''
    # Your code here
    y = year[2:]

    with open('%s_presenters.json' % y, 'r') as f:
        data = json.load(f)

    presenters = json.loads(data)

    return presenters

def pre_ceremony():
    '''This function loads/fetches/processes any data your program
    will use, and stores that data in your DB or in a json, csv, or
    plain text file. It is the first thing the TA will run when grading.
    Do NOT change the name of this function or what it returns.'''
    # Your code here
    # kb and data is a json file

    # Initialize seperate knowledge bases
    people = set()
    media = set()


    ############################
    #### CREATING PEOPLE KB ####
    ############################

    url = 'https://query.wikidata.org/sparql'

    query = """
        # ALL PERSONS required for awards
        SELECT DISTINCT ?person ?personLabel WHERE {
        # FIRST: uncomment occupation:
          ?person wdt:P31 wd:Q5;
                   wdt:P106/wdt:P279* wd:Q2526255; #uncomment for     FILM director (no award for TV director)
          FILTER NOT EXISTS { ?person wdt:P570 ?date. } #person is alive

        # SECOND: uncomment gender if applicable (for actor/actress):
        #          wdt:P21 wd:Q6581097;    #male
        #          wdt:P21 wd:Q6581072;    #female
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """
    r = requests.get(url, params = {'format': 'json', 'query': query})
    kb = r.json()
    for item in kb['results']['bindings']:
        people.add(unidecode.unidecode(item['personLabel']['value']))


    query = """
        # ALL PERSONS required for awards
        SELECT DISTINCT ?person ?personLabel WHERE {
        # FIRST: uncomment occupation:
          ?person wdt:P31 wd:Q5;
                   wdt:P106/wdt:P279* wd:Q10800557; #uncomment for    FILM actor (don't just use actor)
          FILTER NOT EXISTS { ?person wdt:P570 ?date. } #person is alive

        # SECOND: uncomment gender if applicable (for actor/actress):
        #          wdt:P21 wd:Q6581097;    #male
        #          wdt:P21 wd:Q6581072;    #female
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """
    r1 = requests.get(url, params = {'format': 'json', 'query': query})
    kb1 = r1.json()
    for item in kb1['results']['bindings']:
        people.add(unidecode.unidecode(item['personLabel']['value']))


    query = """
        # ALL PERSONS required for awards
        SELECT DISTINCT ?person ?personLabel WHERE {
        # FIRST: uncomment occupation:
          ?person wdt:P31 wd:Q5;
                   wdt:P106/wdt:P279* wd:Q10798782; #uncomment for    TV actor (don't just use actor)
          FILTER NOT EXISTS { ?person wdt:P570 ?date. } #person is alive

        # SECOND: uncomment gender if applicable (for actor/actress):
        #          wdt:P21 wd:Q6581097;    #male
        #          wdt:P21 wd:Q6581072;    #female
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """
    r = requests.get(url, params = {'format': 'json', 'query': query})
    kb = r.json()
    for item in kb['results']['bindings']:
        people.add(unidecode.unidecode(item['personLabel']['value']))



    query = """
        # ALL PERSONS required for awards
        SELECT DISTINCT ?person ?personLabel WHERE {
        # FIRST: uncomment occupation:
          ?person wdt:P31 wd:Q5;
                   wdt:P106/wdt:P279* wd:Q36834; #uncomment for       composer (cannot use songwriter)
          FILTER NOT EXISTS { ?person wdt:P570 ?date. } #person is alive

        # SECOND: uncomment gender if applicable (for actor/actress):
        #          wdt:P21 wd:Q6581097;    #male
        #          wdt:P21 wd:Q6581072;    #female
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """
    r = requests.get(url, params = {'format': 'json', 'query': query})
    kb = r.json()
    for item in kb['results']['bindings']:
        people.add(unidecode.unidecode(item['personLabel']['value']))


    query = """
        # ALL PERSONS required for awards
        SELECT DISTINCT ?person ?personLabel WHERE {
        # FIRST: uncomment occupation:
          ?person wdt:P31 wd:Q5;
                   wdt:P106/wdt:P279* wd:Q28389; #uncomment for       screenwriter
          FILTER NOT EXISTS { ?person wdt:P570 ?date. } #person is alive

        # SECOND: uncomment gender if applicable (for actor/actress):
        #          wdt:P21 wd:Q6581097;    #male
        #          wdt:P21 wd:Q6581072;    #female
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """
    r = requests.get(url, params = {'format': 'json', 'query': query})
    kb = r.json()
    for item in kb['results']['bindings']:
        people.add(unidecode.unidecode(item['personLabel']['value']))


    query = """
        # ALL PERSONS required for awards
        SELECT DISTINCT ?person ?personLabel WHERE {
        # FIRST: uncomment occupation:
          ?person wdt:P31 wd:Q5;
                  wdt:P106/wdt:P279* wd:Q177220; #uncomment for       singer
          FILTER NOT EXISTS { ?person wdt:P570 ?date. } #person is alive

        # SECOND: uncomment gender if applicable (for actor/actress):
        #          wdt:P21 wd:Q6581097;    #male
        #          wdt:P21 wd:Q6581072;    #female
          SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }
        }
        """
    r = requests.get(url, params = {'format': 'json', 'query': query})
    kb = r.json()
    for item in kb['results']['bindings']:
        people.add(unidecode.unidecode(item['personLabel']['value']))



    ###########################
    #### CREATING MEDIA KB ####
    ###########################

    url = 'https://query.wikidata.org/sparql'
    query = """
        SELECT DISTINCT ?itemLabel  WHERE {
         ?item wdt:P31 wd:Q11424. ?item wdt:P577 ?_publication_date. ?item wdt:P136 ?_genre.
         ?_genre rdfs:label ?_genreLabel. BIND(str(YEAR(?_publication_date)) AS ?year)
         FILTER((LANG(?_genreLabel)) = "en")
         FILTER (?_publication_date >= "2012-00-00T00:00:00Z"^^xsd:dateTime && ?_publication_date <= "2019-00-00T00:00:00Z"^^xsd:dateTime )
         SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .} }
        """
    r = requests.get(url, params = {'format': 'json', 'query': query})
    kb_m = r.json()

    for item in kb_m['results']['bindings']:
        media.add(unidecode.unidecode(item['itemLabel']['value']))


    query = """
        SELECT DISTINCT ?itemLabel  WHERE {
          ?item wdt:P31 wd:Q5398426.
          ?item wdt:P580  ?_start
         FILTER (?_start >= "2005-00-00T00:00:00Z"^^xsd:dateTime && ?_start <= "2019-00-00T00:00:00Z"^^xsd:dateTime )
          SERVICE wikibase:label {bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en" .}
        }
        """
    r = requests.get(url, params = {'format': 'json', 'query': query})
    kb_m = r.json()

    for item in kb_m['results']['bindings']:
        media.add(unidecode.unidecode(item['itemLabel']['value']))


    # CLEANING KB'S
    people = set(map(lambda x: x.lower(),people))
    media = set(map(lambda x: x.lower(),media))


    class SetEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, set):
                return list(obj)
            return json.JSONEncoder.default(self, obj)

    json_kb_people = json.dumps(people, cls=SetEncoder)
    json_kb_media = json.dumps(media, cls=SetEncoder)

    with open('people_kb.json', 'w') as outfile:
        json.dump(json_kb_people, outfile)

    with open('media_kb.json', 'w') as outfile:
        json.dump(json_kb_media, outfile)

    print("Pre-ceremony processing complete.")
    return

def main(option = True):
    '''This function calls your program. Typing "python gg_api.py"
    will run this function. Or, in the interpreter, import gg_api
    and then run gg_api.main(). This is the second thing the TA will
    run when grading. Do NOT change the name of this function or
    what it returns.'''
    # Your code here
    df13 = pd.read_json('gg2013.json')
    df15 = pd.read_json('../gg2015.json')
    #df18 = pd.read_json('gg2018.json')
    #df19 = pd.read_json('gg2019.json')

    with open('people_kb.json') as f:
        people_kb = json.load(f)
        people_kb = json.loads(people_kb)

    with open('media_kb.json') as f:
        media_kb = json.load(f)
        media_kb = json.loads(media_kb)

    people_kb = set(people_kb)
    media_kb = set(media_kb)


    # sample_size = 200000
    # if len(df15['text']) > sample_size:
    #     df15 = df15.sample(n=sample_size*2)

    # main returns
    hosts13, awards13, nominees13, winners13, presenters13 = main_exec(OFFICIAL_AWARDS_1315,df13,people_kb,media_kb)
    hosts15, awards15, nominees15, winners15, presenters15 = main_exec(OFFICIAL_AWARDS_1315,df15,people_kb,media_kb)

    if option == True:
        hosts18, awards18, nominees18, winners18, presenters18 = main_exec(OFFICIAL_AWARDS_1819,df18,people_kb,media_kb)
        hosts19, awards19, nominees19, winners19, presenters19 = main_exec(OFFICIAL_AWARDS_1819,df19,people_kb,media_kb)
    # hosts1819, awards1819, nominees1819, winners1819, presenters1819 = main_exec(OFFICIAL_AWARDS_1819,twitter_data,people_kb,media_kb)

    json_hosts13 = json.dumps(hosts13)
    json_awards13 = json.dumps(awards13)
    json_nominees13 = json.dumps(nominees13)
    json_winners13 = json.dumps(winners13)
    json_presenters13 = json.dumps(presenters13)

    json_hosts15 = json.dumps(hosts15)
    json_awards15 = json.dumps(awards15)
    json_nominees15 = json.dumps(nominees15)
    json_winners15 = json.dumps(winners15)
    json_presenters15 = json.dumps(presenters15)

    if option == True:
        json_hosts18 = json.dumps(hosts18)
        json_awards18 = json.dumps(awards18)
        json_nominees18 = json.dumps(nominees18)
        json_winners18 = json.dumps(winners18)
        json_presenters18 = json.dumps(presenters18)

        json_hosts19 = json.dumps(hosts19)
        json_awards19 = json.dumps(awards19)
        json_nominees19 = json.dumps(nominees19)
        json_winners19 = json.dumps(winners19)
        json_presenters19 = json.dumps(presenters19)



    with open('13_hosts.json', 'w') as f:
        json.dump(json_hosts13, f, ensure_ascii=False)
    with open('13_awards.json', 'w') as f:
        json.dump(json_awards13, f, ensure_ascii=False)
    with open('13_nominees.json', 'w') as f:
        json.dump(json_nominees13, f, ensure_ascii=False)
    with open('13_winners.json', 'w') as f:
        json.dump(json_winners13, f, ensure_ascii=False)
    with open('13_presenters.json', 'w') as f:
        json.dump(json_presenters13, f, ensure_ascii=False)

    with open('15_hosts.json', 'w') as f:
        json.dump(json_hosts15, f, ensure_ascii=False)
    with open('15_awards.json', 'w') as f:
        json.dump(json_awards15, f, ensure_ascii=False)
    with open('15_nominees.json', 'w') as f:
        json.dump(json_nominees15, f, ensure_ascii=False)
    with open('15_winners.json', 'w') as f:
        json.dump(json_winners15, f, ensure_ascii=False)
    with open('15_presenters.json', 'w') as f:
        json.dump(json_presenters15, f, ensure_ascii=False)

    if option == True:
        with open('18_hosts.json', 'w') as f:
            json.dump(json_hosts18, f, ensure_ascii=False)
        with open('18_awards.json', 'w') as f:
            json.dump(json_awards18, f, ensure_ascii=False)
        with open('18_nominees.json', 'w') as f:
            json.dump(json_nominees18, f, ensure_ascii=False)
        with open('18_winners.json', 'w') as f:
            json.dump(json_winners18, f, ensure_ascii=False)
        with open('18_presenters.json', 'w') as f:
            json.dump(json_presenters18, f, ensure_ascii=False)

        with open('19_hosts.json', 'w') as f:
            json.dump(json_hosts19, f, ensure_ascii=False)
        with open('19_awards.json', 'w') as f:
            json.dump(json_awards19, f, ensure_ascii=False)
        with open('19_nominees.json', 'w') as f:
            json.dump(json_nominees19, f, ensure_ascii=False)
        with open('19_winners.json', 'w') as f:
            json.dump(json_winners19, f, ensure_ascii=False)
        with open('19_presenters.json', 'w') as f:
            json.dump(json_presenters19, f, ensure_ascii=False)



    return

if __name__ == '__main__':
    main()
