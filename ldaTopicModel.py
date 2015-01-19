# See http://radimrehurek.com/gensim/tut1.html#from-strings-to-vectors
# and https://pypi.python.org/pypi/lda.

import lda
import lda.datasets
import numpy as np
import string
import time

from nltk.corpus import stopwords


def ldaModel(k = 25):
    conversations = open("Conversations.txt")
    
    oldMessageTime = None
    documents = []
    conversationGap = 60 * 60
    pattern = "%Y-%m-%d %H:%M:%S"
    cachedStopWords = stopwords.words("english")
    text = []
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        timestamp = contents[0]
        message = " ".join(contents[2:])
        lowers = message.lower()
        no_punctuation = lowers.translate(None, string.punctuation)
        no_digits = ''.join([word for word in no_punctuation if not word.isdigit()])
        no_stops = [word for word in no_digits.split() if word not in cachedStopWords]
        curMessageTime = int(time.mktime(time.strptime(timestamp, pattern)))
        if not oldMessageTime:
            oldMessageTime = curMessageTime
        diff = curMessageTime - oldMessageTime
        if diff > conversationGap:
            documents.append(text)
            text = []
        text += no_stops
        oldMessageTime = curMessageTime
    
    documents.append(text)
    
    token_count = {}
    
    for document in documents:
        for token in document:
            if token not in token_count:
                token_count[token] = 1
            else:
                token_count[token] += 1
    
    allTokenCounts = token_count.items()
    allTokenCounts.sort(key = lambda tokenCount: tokenCount[1], reverse = True)
    numTokens = len(allTokenCounts)
    topOnePer = 0.01 * numTokens
    topOneCount = allTokenCounts[int(topOnePer)][1]
    # Remove words that appear only once or are extremely frequent.
    for i, document in enumerate(documents):
        documents[i] = [word for word in document
                        if 1 < token_count[word] < topOneCount]
    
    index = 0
    token_index = {}
    vocab = []
    
    for document in documents:
        for token in document:
            if token not in token_index:
                token_index[token] = index
                vocab.append(token)
                index += 1
    
    X = np.zeros((len(documents), len(token_index.keys())))
    
    for i, document in enumerate(documents):
        for token in document:
            j = token_index[token]
            X[i][j] += 1
    
    model = lda.LDA(n_topics = k, n_iter = 1000, random_state = 1)
    model.fit(X)  # model.fit_transform(X) is also available
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 15
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))
    
    doc_topic = model.doc_topic_
    topic_scores = []
    for i in range(0 , len(documents)):
        these_topics = []
        for j, topic_score in enumerate(list(doc_topic[i])):
            these_topics.append((j, topic_score))
        these_topics.sort(key = lambda topic_score: topic_score[1], reverse = True)
        topic_scores.append(these_topics)
    
    return topic_scores


if __name__ == "__main__":
    topic_scores = ldaModel()
