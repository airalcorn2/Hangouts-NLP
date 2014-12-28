# Michael A. Alcorn (airalcorn2@gmail.com)
# See http://www.cs.duke.edu/courses/spring14/compsci290/assignments/lab02.html.

from __future__ import print_function

import nltk
import string
import time

from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

def runClusterAnalysis():
    clusterMessages()
    clusterConversations()
    clusterContiguousMessages()


def clusterMessages(k = 50):
    conversations = open("Conversations.txt")
    
    token_dict = {}
    
    line = conversations.readline().strip()
    messagesDict = {}
    i = 0
    
    while line != "":
        contents = line.split("[SEP]")
        message = " ".join(contents[2:])
        lowers = message.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        token_dict[i] = no_punctuation
        messagesDict[i] = line
        i += 1
        line = conversations.readline().strip()
    
    # This can take some time.
    tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')
    tfs = tfidf.fit_transform(token_dict.values())
    
    km = KMeans(n_clusters = k, init = 'k-means++', max_iter = 5000, n_init = 1)
    
    km.fit(tfs)
    
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    for i in range(k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
    
    clusters = {}
    clusterCounts = {}
    
    for i in range(0, len(km.labels_)):
        label = km.labels_[i]
        if label not in clusters:
            clusters[label] = []
            clusterCounts[label] = 0
        clusters[label].append(messagesDict[i])
        clusterCounts[label] += 1
    
    for i in range(0, len(clusterCounts.keys())):
        print("{0}: {1}".format(i, clusterCounts[i]))
    
    output = open("messageClusters_" + str(k), "w")
    
    for label in range(0, len(clusters.keys())):
        separator = ["#" for i in range(0, 50)]
        separator[25] = str(label)
        print("".join(separator), file = output)
        for message in clusters[label]:
            separator = ["*" for i in range(0, 50)]
            print(message, file = output)
    
    output.close()

def clusterConversations(k = 20):
    
    conversations = open("Conversations.txt")
    
    line = conversations.readline().strip()
    oldMessageTime = None
    conversationsDict = {0: []}
    conversation = 0
    conversationGap = 60 * 60
    pattern = "%Y-%m-%d %H:%M:%S"
    
    while line != "":
        contents = line.split("[SEP]")
        timestamp = contents[0]
        curMessageTime = int(time.mktime(time.strptime(timestamp, pattern)))
        if not oldMessageTime:
            oldMessageTime = curMessageTime
        diff = curMessageTime - oldMessageTime
        if diff > conversationGap:
            conversation += 1
            conversationsDict[conversation] = []
        conversationsDict[conversation].append(line)
        oldMessageTime = curMessageTime
        line = conversations.readline().strip()
    
    token_dict = {}
    
    for i in range(0, len(conversationsDict.keys())):
        conversation = conversationsDict[i]
        
        text = ""
        
        for line in conversation:
            contents = line.split("[SEP]")
            message = " ".join(contents[2:])
            lowers = message.lower()
            no_punctuation = lowers.translate(None, string.punctuation)
            text += " " + no_punctuation
        
        text = text[1:]
        token_dict[i] = text
    
    # This can take some time.
    tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')
    tfs = tfidf.fit_transform(token_dict.values())
    
    km = KMeans(n_clusters = k, init = 'k-means++', max_iter = 5000, n_init = 1)
    
    km.fit(tfs)
    
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    for i in range(k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :10]:
            print(' %s' % terms[ind], end='')
        print()
    
    clusters = {}
    clusterCounts = {}
    
    for i in range(0, len(km.labels_)):
        label = km.labels_[i]
        if label not in clusters:
            clusters[label] = []
            clusterCounts[label] = 0
        clusters[label].append(conversationsDict[i])
        clusterCounts[label] += 1
    
    for i in range(0, len(clusterCounts.keys())):
        print("{0}: {1}".format(i, clusterCounts[i]))
    
    output = open("conversationClusters_" + str(k), "w")
    
    for label in range(0, len(clusters.keys())):
        separator = ["#" for i in range(0, 50)]
        separator[25] = str(label)
        print("".join(separator), file = output)
        for conversation in clusters[label]:
            separator = ["*" for i in range(0, 50)]
            print("".join(separator), file = output)
            for message in conversation:
                print(message, file = output)
    
    output.close()

def clusterContiguousMessages(k = 20):
    conversations = open("Conversations.txt")
    
    token_dict = {}
    messages = {}
    
    line = conversations.readline().strip()
    i = 0
    currentSender = None
    currentMessage = ""
    actualMessage = []
    pattern = "%Y-%m-%d %H:%M:%S"
    gapTime = 60 * 3
    prevTime = None
    
    while line != "":
        contents = line.split("[SEP]")
        message = " ".join(contents[2:])
        timestamp = contents[0]
        sender = contents[1]
        if not currentSender:
            currentSender = sender
            prevTime = int(time.mktime(time.strptime(timestamp, pattern)))
        lowers = message.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        currentTime = int(time.mktime(time.strptime(timestamp, pattern)))
        if sender == currentSender and currentTime - prevTime <= gapTime:
            currentMessage += no_punctuation + " "
            actualMessage += [message]
            line = conversations.readline().strip()
            prevTime = currentTime
            continue
        else:
            token_dict[i] = currentMessage[:-1]
            messages[i] = actualMessage
            i += 1
            currentSender = sender
            currentMessage = no_punctuation + " "
            actualMessage = [message]
            prevTime = currentTime
            line = conversations.readline().strip()
    
    token_dict[i] = currentMessage[:-1]
    messages[i] = actualMessage
    
    # This can take some time.
    tfidf = TfidfVectorizer(tokenizer = tokenize, stop_words = 'english')
    tfs = tfidf.fit_transform(token_dict.values())
    
    # km = KMeans(n_clusters = k, init = 'random', max_iter = 5000, n_init = 1)
    km = KMeans(n_clusters = k, init = 'k-means++', max_iter = 5000, n_init = 1)
    
    km.fit(tfs)
    
    print("Top terms per cluster:")
    order_centroids = km.cluster_centers_.argsort()[:, ::-1]
    terms = tfidf.get_feature_names()
    for i in range(k):
        print("Cluster %d:" % i, end='')
        for ind in order_centroids[i, :20]:
            print(' %s' % terms[ind], end='')
        print()
    
    clusters = {}
    clusterCounts = {}
    
    for i in range(0, len(km.labels_)):
        label = km.labels_[i]
        if label not in clusters:
            clusters[label] = []
            clusterCounts[label] = 0
        clusters[label].append(messages[i])
        clusterCounts[label] += 1
    
    for i in range(0, len(clusterCounts.keys())):
        print("{0}: {1}".format(i, clusterCounts[i]))
    
    output = open("contiguousMessageClusters_" + str(k), "w")
    
    for label in range(0, len(clusters.keys())):
        separator = ["#" for i in range(0, 50)]
        separator[25] = str(label)
        print("".join(separator), file = output)
        for conversation in clusters[label]:
            separator = ["*" for i in range(0, 50)]
            print("".join(separator), file = output)
            for message in conversation:
                print(message, file = output)
    
    output.close()

if __name__ == "__main__":
    clusterMessages()
    clusterConversations()
    clusterContiguousMessages()