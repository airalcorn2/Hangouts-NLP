#!/usr/bin/env python

# See http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html.

from __future__ import print_function

import nltk
import string
import sys
import time
import unicodedata

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()
tbl = dict.fromkeys(i for i in xrange(sys.maxunicode)
                    if unicodedata.category(unichr(i)).startswith('P'))


def stem_tokens(tokens, stemmer):
    return [stemmer.stem(item) for item in tokens]


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    return stem_tokens(tokens, stemmer)


def run_cluster_analysis():
    cluster_messages()
    cluster_conversations()
    cluster_contiguous_messages()


def cluster_messages(k = 50):
    print("Clustering messages...")
    conversations = open("Conversations.txt")
    
    token_dict = {}
    
    messages_dict = {}
    messages = []
    
    for (i, line) in enumerate(conversations):
        line = line.strip().decode("utf-8")
        contents = line.strip().split("[SEP]")
        if len(contents) < 3:
            continue
        
        message = " ".join(contents[2:])
        lowers = message.lower()
        no_punctuation = lowers.translate(tbl)
        token_dict[i] = no_punctuation
        messages_dict[i] = line
        messages.append(i)
    
    # This can take some time.
    tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = "english")
    tfs = tfidf.fit_transform(token_dict.values())
    
    km = KMeans(n_clusters = k, init = "k-means++", max_iter = 5000, n_init = 100)
    km.fit(tfs)
    
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    for i in range(k):
        print("Cluster %d:" % i, end = "")
        for ind in order_centroids[i, :10]:
            print(" %s" % terms[ind], end = "")
        print()
    
    clusters = {}
    cluster_counts = {}
    
    for i in range(0, len(km.labels_)):
        label = km.labels_[i]
        if label not in clusters:
            clusters[label] = []
            cluster_counts[label] = 0
        clusters[label].append(messages_dict[messages[i]])
        cluster_counts[label] += 1
    
    for i in range(0, len(cluster_counts)):
        print("{0}: {1}".format(i, cluster_counts[i]))
    
    output = open("Files/message_clusters_{0}.txt".format(k), "w")
    
    for label in range(0, len(clusters)):
        separator = ["#" for i in range(0, 50)]
        separator[25] = str(label)
        print("".join(separator), file = output)
        for message in clusters[label]:
            separator = ["*" for i in range(0, 50)]
            print(message.encode("utf-8"), file = output)
    
    output.close()


def cluster_conversations(k = 20):
    
    print("Clustering converations...")
    conversations = open("Conversations.txt")
    
    old_message_time = None
    conversations_dict = {0: []}
    conversation = 0
    conversation_gap = 60 * 60
    pattern = "%Y-%m-%d %H:%M:%S"
    
    line = None
    
    for text in conversations:
        line = text
        line = line.strip().decode("utf-8")
        contents = line.strip().split("[SEP]")
        if len(contents) < 3:
            continue
        
        timestamp = contents[0]
        cur_message_time = int(time.mktime(time.strptime(timestamp, pattern)))
        if not old_message_time:
            old_message_time = cur_message_time
        diff = cur_message_time - old_message_time
        if diff > conversation_gap:
            conversation += 1
            conversations_dict[conversation] = []
        conversations_dict[conversation].append(line)
        old_message_time = cur_message_time
    
    conversations_dict[conversation].append(line)
    token_dict = {}
    
    for i in range(0, len(conversations_dict)):
        conversation = conversations_dict[i]
        
        text = ""
        
        for line in conversation:
            contents = line.split("[SEP]")
            message = " ".join(contents[2:])
            lowers = message.lower()
            no_punctuation = lowers.translate(tbl)
            text += " " + no_punctuation
        
        text = text[1:]
        token_dict[i] = text
    
    # This can take some time.
    tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = "english")
    tfs = tfidf.fit_transform(token_dict.values())
    
    km = KMeans(n_clusters = k, init = "k-means++", max_iter = 5000, n_init = 1)
    km.fit(tfs)
    
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    for i in range(k):
        print("Cluster %d:" % i, end = "")
        for ind in order_centroids[i, :10]:
            print(" %s" % terms[ind], end = "")
        print()
    
    clusters = {}
    cluster_counts = {}
    
    for i in range(0, len(km.labels_)):
        label = km.labels_[i]
        if label not in clusters:
            clusters[label] = []
            cluster_counts[label] = 0
        clusters[label].append(conversations_dict[i])
        cluster_counts[label] += 1
    
    for i in range(0, len(cluster_counts)):
        print("{0}: {1}".format(i, cluster_counts[i]))
    
    output = open("Files/conversation_clusters_{0}.txt".format(k), "w")
    
    for label in range(0, len(clusters)):
        separator = ["#" for i in range(0, 50)]
        separator[25] = str(label)
        print("".join(separator), file = output)
        for conversation in clusters[label]:
            separator = ["*" for i in range(0, 50)]
            print("".join(separator), file = output)
            for message in conversation:
                print(message.encode("utf-8"), file = output)
    
    output.close()


def cluster_contiguous_messages(k = 20):
    print("Clustering contiguous messages...")
    conversations = open("Conversations.txt")
    
    token_dict = {}
    messages = {}
    
    i = 0
    current_sender = None
    current_message = ""
    actual_message = []
    pattern = "%Y-%m-%d %H:%M:%S"
    gap_time = 60 * 3
    prev_time = None
    
    for line in conversations:
        line = line.strip().decode("utf-8")
        contents = line.strip().split("[SEP]")
        if len(contents) < 3:
            continue
        
        message = " ".join(contents[2:])
        timestamp = contents[0]
        sender = contents[1]
        if not current_sender:
            current_sender = sender
            prev_time = int(time.mktime(time.strptime(timestamp, pattern)))
        lowers = message.lower()
        no_punctuation = lowers.translate(tbl)
        current_time = int(time.mktime(time.strptime(timestamp, pattern)))
        if sender == current_sender and current_time - prev_time <= gap_time:
            current_message += no_punctuation + " "
            actual_message += [message]
            prev_time = current_time
            continue
        else:
            token_dict[i] = current_message[:-1]
            messages[i] = actual_message
            i += 1
            current_sender = sender
            current_message = no_punctuation + " "
            actual_message = [message]
            prev_time = current_time
    
    token_dict[i] = current_message[:-1]
    messages[i] = actual_message
    
    # This can take some time.
    tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = "english")
    tfs = tfidf.fit_transform(token_dict.values())
    
    # km = KMeans(n_clusters = k, init = 'random', max_iter = 5000, n_init = 1)
    km = KMeans(n_clusters = k, init = "k-means++", max_iter = 5000, n_init = 1)
    km.fit(tfs)
    
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    for i in range(k):
        print("Cluster %d:" % i, end = "")
        for ind in order_centroids[i, :20]:
            print(" %s" % terms[ind], end = "")
        print()
    
    clusters = {}
    cluster_counts = {}
    
    for i in range(0, len(km.labels_)):
        label = km.labels_[i]
        if label not in clusters:
            clusters[label] = []
            cluster_counts[label] = 0
        clusters[label].append(messages[i])
        cluster_counts[label] += 1
    
    for i in range(0, len(cluster_counts)):
        print("{0}: {1}".format(i, cluster_counts[i]))
    
    output = open("Files/contiguous_message_clusters_{0}.txt".format(k), "w")
    
    for label in range(0, len(clusters)):
        separator = ["#" for i in range(0, 50)]
        separator[25] = str(label)
        print("".join(separator), file = output)
        for conversation in clusters[label]:
            separator = ["*" for i in range(0, 50)]
            print("".join(separator), file = output)
            for message in conversation:
                print(message.encode("utf-8"), file = output)
    
    output.close()


def main():
    cluster_messages()
    cluster_conversations()
    cluster_contiguous_messages()


if __name__ == "__main__":
    main()