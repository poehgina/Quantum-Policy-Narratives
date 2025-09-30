# -*- coding: utf-8 -*-
"""
How Nations Narrate Quantum Policy: A Topic Modeling Approach to National Quantum Strategies
Topic Modeling approach
"""

#%%
#importing all libraries necessary
# USING BERTOPIC v0.15

import random
random.seed(42)
import torch
torch.manual_seed(42)

import numpy as np
np.random.seed(42)
import pandas as pd
import time
import json
import plotly.io as pio
pio.renderers.default = 'browser'
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from topictuner import TopicModelTuner as TMT


#%%%
######################################################
#########            Load Data     ######
######################################################

# Replace the file path with the path to your CSV file
csv_file_path = '/dataframe_policy.csv' # insert your working directory to load dataset
# Read the CSV file
txts_df1 = pd.read_csv(csv_file_path)

# Check if Data was imported correctly
#Overview
print(txts_df1)
print(txts_df1.keys())

# Filenumber 
unique_filenames = txts_df1['filename'].unique()
print(unique_filenames)
num_unique_filenames = txts_df1['filename'].nunique()
print(f"Number of unique filenames: {num_unique_filenames}")
#--> 55 documents

######################################################
#########            Descriptives     ######
######################################################

# Sentence count per country
country_counts = txts_df1['country'].value_counts()
print(country_counts) 
# Files per year
unique_files_per_year = txts_df1.groupby('year')['filename'].nunique().reset_index()
unique_files_per_year.columns = ['year', 'Unique Filenames']
print(unique_files_per_year)

# Documents per country
docs_per_country = txts_df1.groupby('country')['filename'].nunique().reset_index()
docs_per_country.columns = ['Country', 'Number of Unique Documents']
print(docs_per_country)

# Overview of countries and years they issued documents
filename_to_docid = {filename: idx+1 for idx, filename in enumerate(sorted(txts_df1['filename'].unique()))}
txts_df1['doc_id'] = txts_df1['filename'].map(filename_to_docid)
print(txts_df1[['filename', 'doc_id']].head())

# Group by country and years
years_by_country = txts_df1.groupby('country')['year'].unique().reset_index()
years_by_country['Years Documents Were Released'] = years_by_country['year'].apply(lambda x: ', '.join(map(str, sorted(x))))
years_by_country = years_by_country.drop(columns=['year'])
years_by_country = years_by_country.rename(columns={'country': 'Issuing Country/Region'})

print(years_by_country)
#%%
######################################################
    #### Preparation for BERTopic ####
######################################################
qdocs = txts_df1.token.values # return the sentences into array so BERTopic can use it

sent_model = SentenceTransformer('all-distilroberta-v1')
sent_model.max_seq_length = 512 # change max sequence length to 512
embeddings = sent_model.encode(qdocs, show_progress_bar=True)

#####Change to your location
with open('/embeddings.npy', 'wb') as f:
    np.save(f, embeddings)
    
embeddings = np.load('/embeddings.npy')

from umap import UMAP
# n_neighbors: default 15, Increasing this value typically results in a more global view of the embedding structure 
# whilst smaller values result in a more local view. Increasing this value often results in larger clusters being created

# n_components: default 5, dimensionality of the embeddings after reducing them.
# Increase this value too much and HDBSCAN will have a hard time clustering.
# Lower this value too much and too little information in the resulting embeddings are available to create proper clusters

# min_dist: default 0.1, how tightly UMAP is allowed to pack points together
# low value useful for clustering, larger value prevents packing points together
# set to 0, since we want to cluster

# set rest as default given in BERTopic doc
umap_model = UMAP(n_neighbors=15, n_components=5, metric='cosine', min_dist=0, random_state=42)

# embeds_reduced = umap_model.fit_transform(embeddings)

from sklearn.feature_extraction.text import CountVectorizer
# ngram range: default 1, advised to keep between 1 & 3, will also consider bigrams and trigrams
# stop_words: prevent stop words from appearing in topic representations
# min_df: default 1, how frequent a word must be before being added to our representation
# value should be smaller for smaller datasets

#%%
# No specific stop phrase removal – Simply lowercase 
def remove_stop_phrases(doc):
    return doc.lower()

class NewVectorizer(CountVectorizer):
    def _word_ngrams(self, tokens, stop_words=None):

        # handle stop words (basic English stopwords like 'the', 'and', etc.)
        if stop_words is not None:
            tokens = [w for w in tokens if w not in stop_words]

        # handle token n-grams
        min_n, max_n = self.ngram_range
        if max_n != 1:
            original_tokens = tokens
            if min_n == 1:
                tokens = list(original_tokens)
                min_n += 1
            else:
                tokens = []

        n_original_tokens = len(original_tokens)

        tokens_append = tokens.append
        space_join = " ".join

        for n in range(min_n, min(max_n + 1, n_original_tokens + 1)):
            for i in range(n_original_tokens - n + 1):
                tokens_append(space_join(original_tokens[i: i + n]))

        return tokens

# Create the vectorizer model
vectorizer_model = NewVectorizer(
    ngram_range=(1, 3),  # unigrams, bigrams, trigrams
    stop_words="english",  # still remove basic English stopwords
    preprocessor=remove_stop_phrases  
)

#%%
# https://pberba.github.io/stats/2020/01/17/hdbscan/
import hdbscan
#from cuml.cluster import HDBSCAN
# min_cluster_size: default 5, BERTopic sets at 10, controls min size of cluster and number of clusters generated
# Increasing this value results in fewer clusters but of larger size 
# decreasing this value results in more micro clusters being generated
# BERTopic advises to increase this value than decrease it. Keep at 10 if small dataset

# min_samples: default equal to min_cluster_size, higher discards more outliers to increase cluster size
# controls the number of outliers generated.
# too small and too noisy
clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=3, metric='euclidean', prediction_data=True)
#hdbscan_model = HDBSCAN(min_samples=10, gen_min_span_tree=True, prediction_data=True)


#%%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

#%%
####################################################
        #### TOPIC MODEL  #####
####################################################

# This documentation first shows the final BERTopic Model and the final data analysis. 
# Parameter optimization etc. have been moved to the back 

####################################################
        #### FINAL TOPIC MODEL  #####
####################################################
# Run Topic Model with refined parameters
starttime = time.time()
topic_model1 = BERTopic(language='english',umap_model=umap_model,embedding_model=sent_model,
                       hdbscan_model=clusterer, vectorizer_model=vectorizer_model, top_n_words=15,
                       calculate_probabilities=True)
topics1, probs1 = topic_model1.fit_transform(qdocs)
endtime = time.time()
print("Time elapsed: {}".format(endtime - starttime)) 
print("Topic -1 occurrence: ",topics1.count(-1))
# 8116 ouliers 

"""
CLOSER LOOK AT TOPICS and overall Model
"""
topic_model1.get_topic_info()
topic_model1.get_representative_docs()
# Generate the hierarchical clustering visualization
hierarchy_fig = topic_model1.visualize_hierarchy()
hierarchy_fig.show()
# Save the plot as an HTML file
hierarchy_fig.write_html("hierarchical_clustering.html")


#%%
######################################################
### GET TOP REPRESENTATIONS  OF THE MODEL ############
######################################################

t_topics = topic_model1.get_topic_info()
t_topics = pd.DataFrame.from_dict(t_topics)

import os
print(os.getcwd())
os.chdir('/Users/ginapohlmann/Desktop/Policy_DATA/') # insert your working directory
t_topics.to_excel('t_topics_det_overview.xlsx', index=False, header=True) # export topics for overview and to screen for similarities to merge similar topics afterwards


t_topics = topic_model1.get_topic_info(-1)
t_topics = pd.DataFrame.from_dict(t_topics)
documents = pd.DataFrame({"Document": qdocs,
                          "ID": range(len(qdocs)),
                          "Topic": topic_model1.topics_})

repr_docs, _, _,_ = topic_model1._extract_representative_docs(c_tf_idf=topic_model1.c_tf_idf_,
                                                          documents=documents,
                                                          topics=topic_model1.topic_representations_ ,
                                                          nr_repr_docs=25)

t_topics = topic_model1.get_topics()
t_topics = pd.DataFrame.from_dict(t_topics)
t_topics.to_csv(r't_topics.csv', index=False, header=True) # extracts only top keyword overview per topic

topic_model1.topic_representations_

sent_topics = pd.DataFrame.from_dict(repr_docs)
print(os.getcwd())
os.chdir('/Users/ginapohlmann/Desktop/Policy_DATA/') # insert your working directory
sent_topics.to_csv('sent_overv.csv', index=False, header=True)

#%%
######################################################
################## Save topic model ##################
######################################################

topics1, probs1 = topic_model1.fit_transform(qdocs)

import json

# Saving BERTopic model
embedding_model = "sentence-transformers/all-distilroberta-v1"
topic_model1.save(path="/Users/ginapohlmann/Desktop/Policy_DATA/BERTopic Model",
                  serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model)
topic_model1 = BERTopic.load("/Users/ginapohlmann/Desktop/Policy_DATA/BERTopic Model") # insert your working directory

# Save topics
    # Save topics to JSON file
topics_file = "topics1.json"

with open("topics1.json", 'w') as f:
    json.dump(topics1, f, indent=2) 
    

    # LOAD topics file 
    with open("topics1.json", 'r') as f:
        topics1 = json.load(f)

    
# Load topics from JSON file
try:
    with open(topics_file, 'r') as f:
        topics1 = json.load(f)  # Corrected to use json.load
except json.JSONDecodeError as e:
    print(f"Failed to decode JSON: {e}")
except ValueError as ve:
    print(f"ValueError: {ve}")
    

# Save probabilities   
np.save('probs1.npy', probs1)
probs1 = np.load('probs1.npy') # LOAD probs1 file
# shows probabilities under respective topics per document.
probs_df=pd.DataFrame(probs1)
probs_df['main percentage'] = pd.DataFrame({'max': probs_df.max(axis=1)})

#%%

######################################################
#########           MERGING TOPICS    ######
######################################################
# After running the Topic Model 1 with 90 topics, in this step we iteratively merge similar topics in several runs (4 in total), after we end up with our final topic model containing 64 topics. 
# Update the topic model after the last run to update topic prevalences and representations of topics.
topics_to_merge = [[-1,20,50,57,81,88],[4,14,68],[12,13],[43,44],
                   [40,48],
                   [51,61],
                   [64,22],
                   [7,66],
                   [52,76],
                   [30,78],
                   [86,42],
                   [75,89]]
topic_model1.merge_topics(qdocs, topics_to_merge)

t_topics = topic_model1.get_topics()
t_topics = pd.DataFrame.from_dict(t_topics)
import os
print(os.getcwd())# insert your working directory
t_topics.to_csv(r'Merged_TopicsV1_3.csv', index=False, header=True)

topics_to_merge2 = [[1,2],[14,72],[51,43]]
topic_model1.merge_topics(qdocs, topics_to_merge2)

topics_to_merge3 = [[28,23],[51,58],[46,47]]
topic_model1.merge_topics(qdocs, topics_to_merge3)

topics_to_merge3 = [[-1, 6]]
topic_model1.merge_topics(qdocs, topics_to_merge3)

topics_to_merge4 = [[14, 8]]
topic_model1.merge_topics(qdocs, topics_to_merge4)

t_topics = topic_model1.get_topics()
t_topics = pd.DataFrame.from_dict(t_topics)

import os
print(os.getcwd())
os.chdir('/Users/')# insert your working directory
t_topics.to_csv(r'Merged_TopicsV1_2.csv', index=False, header=True)

# Update topics and probs each time after merging
topics1 = topic_model1.topics_
probs1 = topic_model1.probabilities_
probs1 = pd.DataFrame.from_dict(probs1)

import os
print(os.getcwd())
os.chdir('/Users/')# insert your working directory
probs1.to_csv(r'probs_1.csv', index=False, header=True)


######################################################
###### CUSTOMIZE FINAL MODEL WITH 64 TOPICS ##########
######################################################
# Create topic Labels
topic_labels = topic_model1.generate_topic_labels(nr_words=3,
                                                 topic_prefix=True,
                                                 word_length=10,
                                                 separator=", ")

# Topic titles were defined based on representative sentences and keywords
topictitle = [
    'Quantum Cryptography',
    'Quantum Education and Awareness',
    'Quantum Algorithms for Quantum Advantage',
    'European Quantum Strategy',
    'State Support of Startups/Venture in the QT Ecosystem',
    'Nordic Quantum Technology Strategies and Cooperation',
    'Quantum Sensing',
    'QTs with Transformative Market Potentials',
    'Australia’s Quantum Strategy',
    'Quantum Materials, Devices, and Industrial Fabrication',
    'Quantum Simulation: Devices and Applications',
    'Canadian Quantum Strategy',
    'Quantum Mechanical Principles',
    'Data Security in the Quantum Era',
    'National Quantum Programs and Government Coordination',
    'Germany’s Position in Quantum Innovation',
    'Hybrid HPC/QCS Systems',
    'QTs and Societal Impacts',
    'Quantum Error Correction and Fault-Tolerant Qubits',
    'Responsible QT',
    'International Standardization and Benchmarking',
    'Neural Nets',
    'Photonics, Lasers, and Cryogenic Systems Integration',
    'Quantum Imaging',
    'Japan’s Quantum Strategy',
    'Software Environments for Quantum Computing',
    'Safeguarding and Promoting Intellectual Property',
    'Limits of Classical Computing Systems',
    'International Collaboration for Quantum Industry Development',
    'UK’s Quantum Computing Ecosystem',
    'Market Development and Economic Impact of Quantum Technologies',
    'Dutch Quantum Innovation Hubs and Global Partnerships',
    'UK Leader in Quantum',
    'Gender Equality',
    'South Africa’s Quantum Strategy',
    'Quantum Metrology',
    'QTs for National Security and Defense',
    'Canada’s Strategy for Research Collaboration',
    'Regulation and Legislation',
    'International Research Collaboration',
    'Noisy Intermediate Scale Quantum Devices',
    'Software/Hardware Platforms for Fault-Tolerant Quantum Systems',
    'National Strategies for Quantum Information Science',
    'Dutch Leadership in Quantum Technology Development',
    'QTs for Chemistry, Energy, Drug Discovery, and Materials Science',
    'Single-Photon Sources and Detectors',
    'Hubs for Industry-Academia Collaboration',
    'Atomic Clocks',
    'Cryogenic Infrastructure for QTs',
    'Switzerland as a Leader in QTs',
    'Ireland’s Quantum Strategy',
    'Second Quantum Revolution',
    'Brain Drain / Talent Retention',
    'Incentivizing Public and Private Collaboration',
    'Korea’s Quantum Strategy',
    'UK Investments and Global Partnerships',
    'European Quantum Communication Infrastructure',
    'Australia Attracting International Talent',
    'Discovering Commercial Use Cases for QTs',
    'Stakeholder Involvement',
    'Approaches to Emerging Challenges',
    'Quantum Simulations for Drug Design',
    'Developing National Research Centers',
    'Supply Chain Issues',
    'Saudi Arabia’s Quantum Strategy'
]

# Update the topic numbers to match
topicnums = list(range(0, 65))

# Assign the labels to the topic model
topic_model1.set_topic_labels(dict(zip(topicnums, topictitle)))

# Check if everything worked correctly 
topic_model1.visualize_topics()

hierarchical_topics = topic_model1.hierarchical_topics(qdocs)
topic_model1.visualize_hierarchy(hierarchical_topics=hierarchical_topics,custom_labels=True)


### GET ALL REP DOCS OF A TOPIC
# Create a dictionary of all documents per topic
testdata = {topic: [] for topic in set(topics1)}
for topic, doc in zip(topics1, qdocs):
    testdata[topic].append(doc)

# Convert to dataframe
allsentencedf = pd.DataFrame.from_dict(testdata, orient='index').transpose()

# Now DROP the outlier topic (-1)
if -1 in allsentencedf.columns:
    allsentencedf = allsentencedf.drop(columns=[-1])

allsentencedf.to_csv(r'/Users/ginapohlmann/Desktop/Policy_DATA/allsentencespolicy.csv', index=False, header=True)# insert your working directory

######################################################
######## Extract Data for Topics per Country ########
######################################################

# Info: The assignment of Topics to each Theme/Narrative was made manually after the qualitative assesment of data. Therefore the occurance of each narrative per country was calculated manually (in percent with Excel)

topics_per_class = topic_model1.topics_per_class(qdocs, 
    classes=txts_df1.country)
topics_per_class.to_csv(r'/Users/ginapohlmann/Desktop/Policy_DATA/topics_per_country.csv', index=False, header=True)# insert your working directory

####### Additional overview #############
# Create a new DataFrame combining topics and country info
df_topics_country = pd.DataFrame({
    'Country': txts_df1['country'],
    'Topic': topics1
})

# Group by Country and Topic and count how many times each Topic appears
country_topic_counts = df_topics_country.groupby(['Country', 'Topic']).size().unstack(fill_value=0)

# Show the resulting table
print(country_topic_counts)

country_topic_percent = country_topic_counts.div(country_topic_counts.sum(axis=1), axis=0) * 100
country_topic_percent.to_excel('/Users/ginapohlmann/Desktop/Policy_DATA/country_topic_percent.xlsx')
country_topic_counts.to_excel('/Users/ginapohlmann/Desktop/Policy_DATA/country_topic_counts.xlsx')



######################################################
######## Extract Data for Topics over Time ########
######################################################


timestamps = pd.to_datetime(txts_df1['year'].astype(int), format="%Y")
topics_over_time = topic_model1.topics_over_time(qdocs, timestamps, nr_bins=None)
topic_model1.visualize_topics_over_time(topics_over_time,title='<b>National Tech Strategies: Topics over Time</b>',
custom_labels=True)

topics_over_time_df = pd.DataFrame(topics_over_time)

######################################################
#######  PREPARE TOPICS OVER TIME FOR EXPORT #########
######################################################
# Extract the year from the 'Timestamp' column
topics_over_time_df['Year'] = topics_over_time_df['Timestamp']

topics_over_time_df[['Year']]

# Group by 'Year' and 'Topic' and sum the 'Frequency'
topic_freq = topics_over_time_df.groupby(['Year', 'Topic'])['Frequency'].sum()

# Pivot the table
topic_freq_pivot = topic_freq.reset_index().pivot(index='Year', columns='Topic', values='Frequency')
# Reset the index to turn the 'Year' back into a column

# Check the actual column names
topic_freq_pivot.columns
topic_freq_pivot = topic_freq_pivot.drop(columns=[-1])
topic_freq_pivot.reset_index(inplace=True)
topic_freq_pivot.to_excel(r'/Users/ginapohlmann/Desktop/Policy_DATA/topics_overtime.xlsx', index=False, header=True)
topic_freq_pivot


#%%
####################################################
        #### TOPIC MODEL TUNING   #####
####################################################

# Second step after you ran your first topic model as a test without parameter fine tuning
# This section is used to find the HBDSCAN parameters

################ Run a Test Model for overview with default settings #############
starttime = time.time()
topic_modeltest = BERTopic(language='english',umap_model=umap_model,embedding_model=sent_model,
                       hdbscan_model=clusterer, vectorizer_model=vectorizer_model, top_n_words=15,
                       calculate_probabilities=True)
topicst, probs = topic_modeltest.fit_transform(qdocs)

endtime = time.time()
print("Time elapsed: {}".format(endtime - starttime)) #  

print("Topic -1 occurrence: ",topicst.count(-1))
#9230 ouliers 
topic_modeltest.get_topic(0)
topic_modeltest.get_representative_docs(0)


 #### TOPIC MODEL TUNER  #####

# Load the precomputed embeddings
embeddings = np.load('/Users/ginapohlmann/Desktop/Policy_DATA/embeddings.npy')
# Wrap BERTopic model with TopicModelTuner
tmt = TMT.wrapBERTopicModel(topic_modeltest)
#tmt.createEmbeddings(qdocs)
# Bypass the createEmbeddings method and use the precomputed embeddings
tmt.embeddings = embeddings  # Directly set the embeddings
# Proceed with the reduction of embeddings
tmt.reduce()
tmt.createVizReduction()
tmt.visualizeEmbeddings(39,31).show() 


# Start basic random search for overview
lastRunResultsDF = tmt.randomSearch([*range(15,150)], [.1, .2, .4,.6,.8,1], iters=60)
tmt.visualizeSearch(lastRunResultsDF).show()


# Refined topic tuning search after running through some parameters from original search
starttime = time.time()
lastRunResultsDF = tmt.pseudoGridSearch([*range(20,115)], [x/100 for x in range(2,50,10)]) 
endtime = time.time()
print("Time elapsed: {}".format(endtime - starttime))
tmt.visualizeSearch(lastRunResultsDF).show()

lastRunResultsDF = tmt.pseudoGridSearch([*range(25,130)], [x/100 for x in range(1,30,5)]) 
lastRunResultsDF = tmt.pseudoGridSearch([*range(30,110)], [x/100 for x in range(1,18,5)]) 
lastRunResultsDF = tmt.pseudoGridSearch([*range(65,110)], [x/100 for x in range(1,7,5)]) #
lastRunResultsDF = tmt.pseudoGridSearch([*range(40, 55)], [x/100 for x in range(1,4,10)]) #
lastRunResultsDF = tmt.pseudoGridSearch([*range(60, 75)], [x/100 for x in range(0.5,1,5)]) #
 
tmt.visualizeSearch(lastRunResultsDF).show()
lastRunResultsDF.to_csv(r'newPseudoGrid_TopicTuningV015.csv', index=False, header=True)

# More refining via simpleSearch
lastRunResultsDF = tmt.simpleSearch([60,61,62,68,70,71,72], [1,1,1,1,1,1,1])
tmt.visualizeSearch(lastRunResultsDF).show()
#%%

#%%
####################################################
      #### TOPIC TUNING FOR HBDSACAN PARAMETERS #####
####################################################

#### Below is last run with suitable values inserted in the final BERT Model
# Check csv tables for extracted topics to decide on values 
starttime = time.time()
minclust = [51, 52, 53, 54]
minsamp = [5]
for i in minclust:
    for j in minsamp:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=i, min_samples=j, metric='euclidean', prediction_data=True)
        tmodel = BERTopic(language='english',umap_model=umap_model,embedding_model=sent_model,
                               hdbscan_model=clusterer, vectorizer_model=vectorizer_model, top_n_words=15,
                               calculate_probabilities=True)
        t1, p1 = tmodel.fit_transform(qdocs)
        t2 = tmodel.get_topics()
        t2 = pd.DataFrame.from_dict(t2)
        t2.to_csv(f't_topics{i}-{j}.csv', index=False, header=True)
endtime = time.time()
print("Time elapsed: {}".format(endtime - starttime)) # about ?? min

#minclust = [59,60,61,62,64, 65, 66, 67, 68]
#minsamp = [18,19,20,21]

#minclust = [28,29,30,31]
#minsamp = [10,11,12,13]

#minclust = [38,39,40,41,42,43,44,45,46]
#minsamp = [3,4,5,6,7,8,9,10,11,12]

#minclust = [59,60,61,62,64, 65, 66, 67, 68]
#minsamp = [18,19,20,21]

#minclust = [65,65,66,66,67, 67, 68, 68, 69, 69, 70,70,79]
#minsamp = [1,3,1,3, 1,4,1,4, 1,4,1,4,1]

#minclust = [65,65, 67,67, 70,70, 79]
#minsamp = [1,3, 1,4, 1,4, 1]

#minclust = [57, 40]
#minsamp = [23, 40]

#minclust = [50,55,60,65]
#minsamp = [10,15,20,25]

#minclust = [31,32,33]
#minsamp = [14,15]

