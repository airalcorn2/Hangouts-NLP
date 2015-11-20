#!/usr/bin/env python

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


def run_sender_analysis(get_sender_probs = False):
    conversations = open("Conversations.txt")
    
    lengths_writer = csv.DictWriter(open("Files/message_lengths.csv", "w"), ["length", "sender"])
    lengths_writer.writeheader()
    
    hours_writer = csv.DictWriter(open("Files/message_hours.csv", "w"), ["hour", "sender"])
    hours_writer.writeheader()
    
    vocabularies = {}
    
    # For sender classifier.
    sender_features = {}
    
    for line in conversations:
        line = line.strip().decode("utf-8")
        contents = line.split("[SEP]")
        if len(contents) < 3:
            continue
        
        sender = contents[1].replace(",", "")
        
        timestamp = contents[0]
        [date, time] = timestamp.split()
        [hour, minute, second] = time.split(":")
        hours_row = {"hour": hour, "sender": sender}
        hours_writer.writerow(hours_row)
        
        message = " ".join(contents[2:])
        tokens = nltk.word_tokenize(message)
        message_length = len(tokens)
        
        vocabularies[sender] = vocabularies.get(sender, set()) | {token for token in tokens}
        
        lengths_row = {"length": message_length, "sender": sender}
        lengths_writer.writerow(lengths_row)
        
        feats = word_feats(tokens)
        sender_features[sender] = sender_features.get(sender, []) + [(feats, sender)]
    
    # Vocabulary sizes.
    for sender in vocabularies:
        print(sender + " used {0} unique tokens.".format(len(vocabularies[sender])))
    
    run_sender_classifier(sender_features)


def run_sender_classifier(sender_features, get_sender_probs = True, check_sender_convergence = True):
    
    print("Training sender classifier...")
    
    for sender in sender_features:
        random.shuffle(sender_features[sender])
    
    train_cutoff = 0.75
    cutoffs = {}
    for sender in sender_features:
        cutoffs[sender] = int(math.floor(len(sender_features[sender]) * train_cutoff))
    
    train_features = []
    for sender in sender_features:
        train_features += sender_features[sender][:cutoffs[sender]]
    
    test_features = []
    for sender in sender_features:
        test_features += sender_features[sender][cutoffs[sender]:]
    
    classifier = NaiveBayesClassifier.train(train_features)
    classifier.show_most_informative_features(100)
    
    reference = collections.defaultdict(set)
    gold = []
    pred = collections.defaultdict(set)
    preds = []
    
    for (i, (features, label)) in enumerate(test_features):
        gold.append(label)
        reference[label].add(i)
        prediction = classifier.classify(features)
        pred[prediction].add(i)
        preds.append(prediction)
    
    cm = nltk.metrics.ConfusionMatrix(gold, preds)
    print(cm.pretty_format(sort_by_count = True, show_percents = True, truncate = 9))
    print(classification_report(y_true = gold, y_pred = preds))
    print("train on {0} instances, test on {1} instances".format(len(train_features), len(test_features)))
    print("accuracy: {0}".format(nltk.classify.util.accuracy(classifier, test_features)))
    
    # Get sender probabilities of messages.
    if get_sender_probs:
        go_get_sender_probs(classifier, sender_features.keys())
    
    # Check if senders get more or less similar over time.
    if check_sender_convergence:
        go_check_sender_convergence(classifier)


def go_get_sender_probs(classifier, senders):
    
    print("Getting sender/message probabilities...")
    
    conversations = open("Conversations.txt")
    
    output = open("Files/message_probs.txt", "w")
    
    for line in conversations:
        
        line = line.strip().decode("utf-8")
        contents = line.split("[SEP]")
        if len(contents) < 3:
            continue
        
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        tokens = nltk.word_tokenize(message)
        feats = word_feats(tokens)
        probs = classifier.prob_classify(feats)
        print(sender.encode("utf-8") + ": " + message.encode("utf-8"), file = output)
        results = ""
        for each_sender in senders:
            results += each_sender + ": " + str(probs.prob(each_sender)) + "| "
        
        results = results[:-2]
        print(results, file = output)


def go_check_sender_convergence(classifier, number_of_phases = 5):
    
    print("Checking for sender convergence...")
    
    conversations = open("Conversations.txt")

    all_features = []
    timestamps = []
    
    for line in conversations:
        
        line = line.strip().decode("utf-8")
        contents = line.split("[SEP]")
        if len(contents) < 3:
            continue
        
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        timestamp = contents[0]
        tokens = nltk.word_tokenize(message)
        feats = word_feats(tokens)
        all_features.append((feats, sender))
        timestamps.append(timestamp)
    
    group_size = (len(all_features) / number_of_phases) + 1
    phases = []
    for i in range(0, number_of_phases):
        start = i * group_size
        end = start + group_size
        phases.append(all_features[start:end])
    
    phase_count = 1
    for phase in phases:
        
        sender_features = {}
        
        for feature in phase:
            sender = feature[1]
            sender_features[sender] = sender_features.get(sender, []) + [(feature[0], sender)]
        
        for sender in sender_features:
            random.shuffle(sender_features[sender])
        
        train_cutoff = 0.75
        cutoffs = {}
        for sender in sender_features:
            cutoffs[sender] = int(math.floor(len(sender_features[sender]) * train_cutoff))
        
        train_features = []
        for sender in sender_features:
            train_features += sender_features[sender][:cutoffs[sender]]
        
        test_features = []
        for sender in sender_features:
            test_features += sender_features[sender][cutoffs[sender]:]
        
        classifier = NaiveBayesClassifier.train(train_features)
        classifier.show_most_informative_features(100)
        
        reference = collections.defaultdict(set)
        gold = []
        pred = collections.defaultdict(set)
        preds = []
        
        for (i, (features, label)) in enumerate(test_features):
            gold.append(label)
            reference[label].add(i)
            prediction = classifier.classify(features)
            pred[prediction].add(i)
            preds.append(prediction)
        
        cm = nltk.metrics.ConfusionMatrix(gold, preds)
        print(cm.pretty_format(sort_by_count = True, show_percents = True, truncate = 9))
        print(classification_report(y_true = gold, y_pred = preds))
        print("train on {0} instances, test on {1} instances".format(len(train_features), len(test_features)))
        print("accuracy: {0}".format(nltk.classify.util.accuracy(classifier, test_features)))
        
        phase_count += 1


def main():
    run_sender_analysis()


if __name__ == "__main__":
    main()