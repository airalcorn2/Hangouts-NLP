# Michael A. Alcorn (airalcorn2@gmail.com)

from __future__ import print_function

import collections
import csv
import math
import nltk
import nltk.classify.util
import random

from nltk.classify import NaiveBayesClassifier
from sklearn.metrics import classification_report

def word_feats(words):
    return dict([(word, True) for word in words])


def runSenderAnalysis(getSenderProbs = False):
    conversations = open("Conversations.txt")
    
    lengthsWriter = csv.DictWriter(open("Files/messageLengths.csv", "w"), ["length", "sender"])
    lengthsWriter.writeheader()
    
    hoursWriter = csv.DictWriter(open("Files/messageHours.csv", "w"), ["hour", "sender"])
    hoursWriter.writeheader()
    
    vocabularies = {}
    
    # For sender classifier.
    senderFeatures = {}
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        sender = contents[1].replace(",", "")
        
        timestamp = contents[0]
        [date, time] = timestamp.split()
        [hour, minute, second] = time.split(":")
        hoursRow = {"hour": hour, "sender": sender}
        hoursWriter.writerow(hoursRow)
        
        message = " ".join(contents[2:])
        tokens = nltk.word_tokenize(message)
        messageLength = len(tokens)
        
        tokens = nltk.word_tokenize(message)
        if sender not in vocabularies:
            vocabularies[sender] = {}
        
        for token in tokens:
            if token not in vocabularies[sender]:
                vocabularies[sender][token] = True
        
        lengthsRow = {"length": messageLength, "sender": sender}
        lengthsWriter.writerow(lengthsRow)
        
        if sender not in senderFeatures:
            senderFeatures[sender] = []
        
        feats = word_feats(tokens)
        senderFeatures[sender].append((feats, sender))
    
    # Vocabulary sizes.
    for sender in vocabularies.keys():
        print(sender + " used {0} unique tokens.".format(len(vocabularies[sender].keys())))
    
    runSenderClassifier(senderFeatures)


def runSenderClassifier(senderFeatures, getSenderProbs = True, checkSenderConvergence = True):
    
    print("Training sender classifier...")
    
    for sender in senderFeatures.keys():
        random.shuffle(senderFeatures[sender])
    
    trainCutoff = 0.75
    cutoffs = {}
    for sender in senderFeatures.keys():
        cutoffs[sender] = int(math.floor(len(senderFeatures[sender]) * trainCutoff))
    
    trainFeatures = []
    for sender in senderFeatures.keys():
        trainFeatures += senderFeatures[sender][:cutoffs[sender]]
    
    testFeatures = []
    for sender in senderFeatures.keys():
        testFeatures += senderFeatures[sender][cutoffs[sender]:]
    
    classifier = NaiveBayesClassifier.train(trainFeatures)
    classifier.show_most_informative_features(100)
    
    reference = collections.defaultdict(set)
    gold = []
    pred = collections.defaultdict(set)
    preds = []
    
    for i, (features, label) in enumerate(testFeatures):
        gold.append(label)
        reference[label].add(i)
        prediction = classifier.classify(features)
        pred[prediction].add(i)
        preds.append(prediction)
    
    cm = nltk.metrics.ConfusionMatrix(gold, preds)
    print(cm.pp(sort_by_count = True, show_percents = True, truncate = 9))
    print(classification_report(y_true = gold, y_pred = preds))
    print('train on {0} instances, test on {1} instances'.format(len(trainFeatures), len(testFeatures)))
    print('accuracy: {0}'.format(nltk.classify.util.accuracy(classifier, testFeatures)))
    
    # Get sender probabilities of messages.
    if getSenderProbs:
        goGetSenderProbs(classifier, senderFeatures.keys())
    
    # Check if senders get more or less similar over time.
    if checkSenderConvergence:
        goCheckSenderConvergence(classifier)


def goGetSenderProbs(classifier, senders):
    
    print("Getting sender/message probabilities...")
    
    conversations = open("Conversations.txt")
    
    output = open("Files/messageProbs.txt", "w")
    
    for line in conversations:
        
        contents = line.strip().split("[SEP]")
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        tokens = nltk.word_tokenize(message)
        feats = word_feats(tokens)
        probs = classifier.prob_classify(feats)
        print(sender + ": " + message, file = output)
        results = ""
        for eachSender in senders:
            results += eachSender + ": " + str(probs.prob(eachSender)) + "| "
        
        results = results[:-2]
        print(results, file = output)


def goCheckSenderConvergence(classifier, numberOfPhases = 5):
    
    print("Checking for sender convergence...")
    
    conversations = open("Conversations.txt")

    allFeatures = []
    timestamps = []
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        timestamp = contents[0]
        tokens = nltk.word_tokenize(message)
        feats = word_feats(tokens)
        allFeatures.append((feats, sender))
        timestamps.append(timestamp)
    
    groupSize = (len(allFeatures) / numberOfPhases) + 1
    phases = []
    for i in range(0, numberOfPhases):
        start = i * groupSize
        end = start + groupSize
        phases.append(allFeatures[start:end])
    
    phaseCount = 1
    for phase in phases:
        
        senderFeatures = {}
        
        for feature in phase:
            sender = feature[1]
            if sender not in senderFeatures:
                senderFeatures[sender] = []
            
            senderFeatures[sender].append((feature[0], sender))
        
        for sender in senderFeatures.keys():
            random.shuffle(senderFeatures[sender])
        
        trainCutoff = 0.75
        cutoffs = {}
        for sender in senderFeatures.keys():
            cutoffs[sender] = int(math.floor(len(senderFeatures[sender]) * trainCutoff))
        
        trainFeatures = []
        for sender in senderFeatures.keys():
            trainFeatures += senderFeatures[sender][:cutoffs[sender]]
        
        testFeatures = []
        for sender in senderFeatures.keys():
            testFeatures += senderFeatures[sender][cutoffs[sender]:]
        
        classifier = NaiveBayesClassifier.train(trainFeatures)
        classifier.show_most_informative_features(100)
        
        reference = collections.defaultdict(set)
        gold = []
        pred = collections.defaultdict(set)
        preds = []
        
        for i, (features, label) in enumerate(testFeatures):
            gold.append(label)
            reference[label].add(i)
            prediction = classifier.classify(features)
            pred[prediction].add(i)
            preds.append(prediction)
        
        cm = nltk.metrics.ConfusionMatrix(gold, preds)
        print(cm.pp(sort_by_count = True, show_percents = True, truncate = 9))
        print(classification_report(y_true = gold, y_pred = preds))
        print('train on {0} instances, test on {1} instances'.format(len(trainFeatures), len(testFeatures)))
        print('accuracy: {0}'.format(nltk.classify.util.accuracy(classifier, testFeatures)))
        
        phaseCount += 1

if __name__ == "__main__":
    runSenderAnalysis()