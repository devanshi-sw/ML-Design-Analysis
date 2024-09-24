
"""Gibbs Sampling

from google.colab import files

uploaded = files.upload()

import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import gensim
from gensim 
import corpora
import nltk from nltk.stem import WordNetLemmatizer, SnowballStemmer
from gensim.models.ldamulticore import LdaMulticore
from wordcloud import WordCloud
from gensim.models import CoherenceModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score
from google.colab import files

# File named in this project 'designtextdata.csv'
df = pd.read_csv(io.BytesIO(uploaded['designtextdata.csv']))
print(df.head())

# Text Processing Functions
nltk.download('wordnet')

from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem.porter import *

#  lemmatize and stem preprocessing steps 
def lemmatize_stemming(text):
    stemmer = SnowballStemmer('english')
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v'))

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:
            result.append(lemmatize_stemming(token))
    return result

# Preprocess the data
processed_data = df['Problemstatements'].map(preprocess)

# Check processed data
print(processed_data.head())

df['processed_data'] = processed_data
df.to_csv('preprocessed_data.csv', index=False)

# Remove additional stopwords
processed_data = processed_data.apply(lambda x: [word for word in x if word not in additional_stopwords])

from gensim import corpora

# Create Dictionary
id2word = corpora.Dictionary(processed_data)

# Create Corpus
texts = processed_data

# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in texts]

# Build the LDA model
from gensim.models.ldamulticore import LdaMulticore

lda_model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=20, passes=10, workers=2)

# Print the top topics
for idx, topic in lda_model.print_topics(-1):
    print('Topic: {} \nWords: {}'.format(idx, topic))

def display_topics(topics):
    for topic in topics:
        print(f"Topic {topic['Topic']}:")
        for word, prob in topic["Words"].items():
            print(f"{word}: {prob}")
        print()

display_topics(topics)

# Get the dominant topic for each document
def get_dominant_topic(lda_model, corpus, texts):
    dominant_topics_df = pd.DataFrame()

    for i, row in enumerate(lda_model[corpus]):
        row = sorted(row, key=lambda x: (x[1]), reverse=True)
        for j, (topic_num, prop_topic) in enumerate(row):
            if j == 0:  # Dominant topic
                wp = lda_model.show_topic(topic_num)
                topic_keywords = ", ".join([word for word, prop in wp])
                dominant_topics_df = dominant_topics_df.append(pd.Series([int(topic_num), round(prop_topic, 4), topic_keywords]), ignore_index=True)
            else:
                break

    dominant_topics_df.columns = ['Dominant_Topic', 'Perc_Contribution', 'Topic_Keywords']
    contents = pd.Series(texts)
    dominant_topics_df = pd.concat([dominant_topics_df, contents], axis=1)
    return dominant_topics_df

df_topic_keywords = get_dominant_topic(lda_model, corpus, df['Problemstatements'])

# Display random documents under each topic
for topic_num in range(lda_model.num_topics):
    print(f"Sample documents for topic #{topic_num}:\n")
    sample_docs = df_topic_keywords[df_topic_keywords['Dominant_Topic'] == topic_num]['Problemstatements'].sample(3)
    for doc in sample_docs:
        print(f"- {doc}\n")
    print("-"*50)

# Top words for each topic in a DataFrame
def topics_to_dataframe(lda_model):
    all_topics = lda_model.get_topics()
    topic_keywords_df = pd.DataFrame(all_topics, columns=id2word.values())
    return topic_keywords_df

df_topics = topics_to_dataframe(lda_model)
print(df_topics)

def classify_new_documents(lda_model, id2word, new_documents):
    # Preprocess the new documents
    new_processed_docs = [preprocess(doc) for doc in new_documents]

    # Convert processed documents to BoW representation
    new_corpus = [id2word.doc2bow(text) for text in new_processed_docs]

    # Get topics for new documents
    dominant_topics = get_dominant_topic(lda_model, new_corpus, new_documents)

    return dominant_topics

new_docs = ["This is a new document about engineering.", "Another document about a different topic."]
results = classify_new_documents(lda_model, id2word, new_docs)
print(results)

!pip install pandas==1.3.3 gensim==4.1.2 nltk==3.6.3
!pip install scikit-learn

pip install pyLDAvis==3.3.1 --no-deps

!pip install funcy

import pyLDAvis.gensim_models as gensimvis
import pyLDAvis



# Prepare the visualization data
vis = pyLDAvis.gensim_models.prepare(lda_model, corpus, id2word, mds='mmds', n_jobs=1)

# Visualize the topics
pyLDAvis.enable_notebook()
vis

pip install wordcloud

import matplotlib.pyplot as plt
from wordcloud import WordCloud

def generate_wordclouds(lda_model, num_topics):
    # Extracting topics from the LDA model
    topics = lda_model.show_topics(num_topics=num_topics, formatted=False)

    for topic, keywords in topics:
        # Generating the word cloud
        wc = WordCloud(
            background_color='white',
            max_words=100,
            width=600,
            height=400,
            prefer_horizontal=1.0,
            scale=0.5,
            colormap='viridis'
        )

        topic_words = dict(keywords)
        wc.generate_from_frequencies(topic_words)

        # Plotting
        plt.figure(figsize=(10, 7))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f"Topic {topic}")
        plt.show()

# Assuming lda_model is already defined elsewhere in your code
generate_wordclouds(lda_model, num_topics=lda_model.num_topics)

from gensim.models import CoherenceModel

# Compute Perplexity
print('\nPerplexity: ', lda_model.log_perplexity(corpus))  # A measure of how good the model is; lower is better.

# Compute Coherence Score
coherence_model_lda = CoherenceModel(model=lda_model, texts=texts, dictionary=id2word, coherence='c_v')
coherence_lda = coherence_model_lda.get_coherence()
print('\nCoherence Score: ', coherence_lda)

import matplotlib.pyplot as plt

def compute_perplexity_for_topics(start=2, end=30, step=1):
    num_topics = []
    perplexities = []

    for topics in range(start, end+1, step):
        lda_model = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=topics, passes=10, workers=2)
        perplexity = lda_model.log_perplexity(corpus)
        num_topics.append(topics)
        perplexities.append(perplexity)
        print(f"Number of Topics: {topics}, Perplexity Score: {perplexity}")

    return num_topics, perplexities

num_topics, perplexities = compute_perplexity_for_topics()

plt.figure(figsize=(10, 7))
plt.plot(num_topics, perplexities, '-o')
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity")
plt.title("Perplexity vs. Number of Topics")
plt.show()

from gensim.models import CoherenceModel

# Function to compute coherence values
def compute_coherence_values(dictionary, corpus, texts, limit, start=2, step=3):
    coherence_values = []
    model_list = []
    for num_topics in range(start, limit, step):
        model = LdaMulticore(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10, workers=2)
        model_list.append(model)
        coherencemodel = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        coherence_values.append(coherencemodel.get_coherence())
    return model_list, coherence_values

# Compute coherence values
model_list, coherence_values = compute_coherence_values(dictionary=id2word, corpus=corpus, texts=processed_data, start=2, limit=40, step=6)

# Plot coherence score values
import matplotlib.pyplot as plt
limit=40; start=2; step=6;
x = range(start, limit, step)
plt.plot(x, coherence_values)
plt.xlabel("Num Topics")
plt.ylabel("Coherence score")
plt.legend(("coherence_values"), loc='best')
plt.show()

# Example for tuning alpha and beta parameters
alphas = list(np.arange(0.01, 1, 0.3))
betas = list(np.arange(0.01, 1, 0.3))
num_topics = [10, 15, 20]

best_model = None
best_coherence = -1
best_params = {}

for alpha in alphas:
    for beta in betas:
        for topic in num_topics:
            lda = LdaMulticore(corpus=corpus, id2word=id2word, num_topics=topic, passes=10, alpha=alpha, eta=beta, workers=2)
            coherence_model_lda = CoherenceModel(model=lda, texts=processed_data, dictionary=id2word, coherence='c_v')
            coherence = coherence_model_lda.get_coherence()
            if coherence > best_coherence:
                best_coherence = coherence
                best_model = lda
                best_params = {'Alpha': alpha, 'Beta': beta, 'Num_Topics': topic}

print("Best Model's Params: ", best_params)
print("Best Coherence Score: ", best_coherence)



from gensim.models import HdpModel
# HDP Model
id2word.filter_extremes(no_below=15, no_above=0.5)
corpus = [id2word.doc2bow(text) for text in texts]
hdp = HdpModel(corpus=corpus, id2word=id2word, alpha=10, gamma=1, T=20)
 # Assuming 'texts' is your tokenized text data


# HDP Model
hdp = HdpModel(corpus=corpus, id2word=id2word)

# Print the topics produced by HDP
topics = hdp.print_topics(num_topics=10)  # Adjust num_topics as needed
for topic in topics:
    print(topic)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# Convert the documents into vectors using TfidfVectorizer
vectorizer = TfidfVectorizer(vocabulary=vocab_list)
X = vectorizer.transform([' '.join(doc) for doc in processed_data])

# Convert cluster assignments to labels for silhouette_score
labels = mgp.choose_best_label(processed_data)

# Compute silhouette score
sil_score = silhouette_score(X.toarray(), labels)
print(f"Silhouette Score: {sil_score}")

from gensim.models.coherencemodel import CoherenceModel

# Extract top words for each topic from HDP
top_words_hdp = []
for _, topic in topics:
    # Splitting the string and extracting words, e.g., '0.003*heat + 0.003*assist' -> ['heat', 'assist']
    top_words = [word.split('*')[1].strip() for word in topic.split('+')]
    top_words_hdp.append(top_words)

# Calculate coherence score
coherence_hdp = CoherenceModel(topics=top_words_hdp, texts=processed_data, dictionary=id2word, coherence='c_v').get_coherence()

print("HDP Coherence:", coherence_hdp)

num_topics_discovered = len(hdp.get_topics())
print(f"Number of topics discovered by HDP: {num_topics_discovered}")

from gsdmm import MovieGroupProcess

# Define GSDMM parameters
K = 10  # You can change this based on your initial guess or using insights from LDA
alpha = 0.1
beta = 0.1
n_iters = 25  # number of iterations, you can adjust based on your requirements

mgp = MovieGroupProcess(K=K, alpha=alpha, beta=beta, n_iters=n_iters)

vocab = set(x for doc in processed_data for x in doc)
n_terms = len(vocab)
n_docs = len(processed_data)

# Fit the model
y = mgp.fit(processed_data, n_terms)

# Get the topics
doc_count = np.array(mgp.cluster_doc_count)
print('Number of documents per topic :', doc_count)
print('*'*20)

# Topics sorted by the number of document they are allocated to
top_index = doc_count.argsort()[-10:][::-1]
print('Most important clusters (by number of docs inside):', top_index)
print('*'*20)

# Convert set vocabulary to a list for index retrieval
vocab_list = list(vocab)

# Print top words for each cluster
for i in range(len(top_index)):
    word_freq = mgp.cluster_word_distribution[top_index[i]]
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    top_words = [word for word, _ in sorted_word_freq]
    print('Cluster {}: {}'.format(top_index[i], top_words))

# Extract top words for each cluster from MGP
top_words_mgp = []
for i in range(K):  # K is the number of topics
    word_freq = mgp.cluster_word_distribution[i]
    sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]  # top 5 words, adjust as needed
    top_words = [word for word, _ in sorted_word_freq]
    top_words_mgp.append(top_words)



from gensim.models import CoherenceModel

coherence_model_mgp = CoherenceModel(topics=top_words_mgp, texts=processed_data, dictionary=id2word, coherence='c_v')
coherence_mgp = coherence_model_mgp.get_coherence()
print("MGP Coherence:", coherence_mgp)

sample_labels = [mgp.choose_best_label(doc) for doc in processed_data[:5]]
for label in sample_labels:
    print(label)

labels = [mgp.choose_best_label(doc)[0] for doc in processed_data]

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# Convert the documents into vectors using TfidfVectorizer
vectorizer = TfidfVectorizer(vocabulary=vocab_list)
# Fit the vectorizer to your data
vectorizer.fit([' '.join(doc) for doc in processed_data])
# Now, transform your data
X = vectorizer.transform([' '.join(doc) for doc in processed_data])


# Adjust the labels to extract only the cluster ID
labels = [mgp.choose_best_label(doc)[0] for doc in processed_data]

# Compute the silhouette score
sil_score = silhouette_score(X.toarray(), labels)
print(f"Silhouette Score: {sil_score}")

import matplotlib.pyplot as plt
import seaborn as sns


# Using get() method with dictionary to handle words that might not be present.
# It will return 0 for words that are not found in the dictionary.

for cluster, words in clusters.items():
    frequencies = [word_freq.get(word, 0) for word in words]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=words, y=frequencies)
    plt.title(cluster)
    plt.show()

from wordcloud import WordCloud

for cluster, words in clusters.items():
    wordcloud = WordCloud(background_color="white").generate(" ".join(words))
    plt.figure(figsize=(10, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.title(cluster)
    plt.axis("off")
    plt.show()

# Assuming you have a list or dict with the number of documents per cluster
doc_per_cluster = {'Cluster 8': 61, 'Cluster 9': 36, 'Cluster 0': 21, # ... continue this for all clusters
}

plt.figure(figsize=(10, 6))
sns.barplot(x=list(doc_per_cluster.keys()), y=list(doc_per_cluster.values()))
plt.xticks(rotation=45)
plt.title("Number of Documents per Cluster")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns


from gensim import corpora, models

# Assuming `documents` is a list of lists containing tokenized words from your data
dictionary = corpora.Dictionary(documents)
corpus = [dictionary.doc2bow(text) for text in documents]

lda = models.LdaModel(corpus, id2word=dictionary, num_topics=10, iterations=1000, passes=15)
topics = lda.show_topics(formatted=False, num_words=5)

for idx, topic in topics:
    words = [word[0] for word in topic]
    frequencies = [word[1] for word in topic]
    plt.figure(figsize=(10, 6))
    sns.barplot(x=words, y=frequencies)
    plt.title(f"Topic {idx}")
    plt.show()
