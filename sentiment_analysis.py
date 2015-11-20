#!/usr/bin/env python

from __future__ import print_function

import csv
import nltk
import sys
import time

from nltk.classify import NaiveBayesClassifier


def word_feats(words):
    return dict([(word, True) for word in words])


def run_sentiment_analysis(check_contiguous_messages = True, check_conversations = True, check_weeks = True):
    
    print("Performing sentiment analysis...")
    
    f = open("sentiment_training.txt")
    tweets = [line.strip().decode("utf-8", "ignore") for line in f]
    total_tweets = len(tweets)
    
    features = []
    
    count = 0
    
    # Takes a while.
    for tweet in tweets:
        if count % 10000 == 0:
            percent = 100 * count / total_tweets
            sys.stdout.write("\rTraining {0}% complete...".format(percent))
            sys.stdout.flush()
        [sentiment, sentence] = tweet.split("[SEP]")
        pos_or_neg = None
        if sentiment == "0":
            pos_or_neg = "neg"
        elif sentiment == "4":
            pos_or_neg = "pos"
        else:
            continue
        tokens = nltk.word_tokenize(sentence)
        feats = word_feats(tokens)
        features.append((feats, pos_or_neg))
        count += 1
    
    print()
    classifier = NaiveBayesClassifier.train(features)
    
    classifier.show_most_informative_features(100)
    
    message_sentiment_analysis(classifier)
    if check_contiguous_messages:
        contiguous_messages_sentiment_analysis(classifier)
    
    if check_conversations:
        conversation_sentiment_analysis(classifier)
    
    if check_weeks:
        weekly_sentiment_analysis(classifier)


def message_sentiment_analysis(classifier):
    print("Getting message sentiment scores...")
    
    conversations = open("Conversations.txt")
    
    fieldnames = ["sentiment", "sender"]
    writer = csv.DictWriter(open("Files/message_sentiments.csv", "w"), fieldnames)
    writer.writeheader()
    
    sentiment_scores = []
    
    # Individual messages.
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
        pos = probs.prob("pos")
        row = {"sentiment": pos, "sender": sender}
        writer.writerow(row)
        sentiment_scores.append((pos, message))
    
    sentiment_scores.sort(key = lambda message: message[0], reverse = True)
    
    output = open("Files/message_sentiment_scores.txt", "w")
    for message in sentiment_scores:
        print("{0}: {1}".format(message[0], message[1].encode("utf-8")), file = output)
    
    output.close()


def contiguous_messages_sentiment_analysis(classifier):
    print("Getting contiguous messages sentiment scores...")
    
    conversations = open("Conversations.txt")
    
    fieldnames = ["sentiment", "sender"]
    writer = csv.DictWriter(open("Files/contiguous_message_sentiments.csv", "w"), fieldnames)
    writer.writeheader()
    
    sentiment_scores = []
    current_sender = None
    current_message = ""
    current_tokens = []
    pattern = "%Y-%m-%d %H:%M:%S"
    gap_time = 60 * 10
    prev_time = None
    
    # Contiguous messages.
    for line in conversations:
        line = line.strip().decode("utf-8")
        contents = line.split("[SEP]")
        if len(contents) < 3:
            continue
        
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        timestamp = contents[0]
        if not current_sender:
            current_sender = sender
            prev_time = int(time.mktime(time.strptime(timestamp, pattern)))
        tokens = nltk.word_tokenize(message)
        current_time = int(time.mktime(time.strptime(timestamp, pattern)))
        if sender == current_sender and current_time - prev_time <= gap_time:
            current_tokens += tokens
            current_message += message + " "
            prev_time = current_time
            continue
        else:
            feats = word_feats(current_tokens)
            probs = classifier.prob_classify(feats)
            pos = probs.prob("pos")
            row = {"sentiment": pos, "sender": sender}
            writer.writerow(row)
            sentiment_scores.append((pos, current_message[:-1]))
            current_sender = sender
            prev_time = current_time
            current_message = message + " "
            current_tokens = tokens
    
    # Last batch.
    feats = word_feats(current_tokens)
    probs = classifier.prob_classify(feats)
    pos = probs.prob("pos")
    row = {"sentiment": pos, "sender": sender}
    writer.writerow(row)
    
    sentiment_scores.append((pos, current_message[:-1]))
    
    sentiment_scores.sort(key = lambda message: message[0], reverse = True)
    
    output = open("Files/contiguous_message_sentiment_scores.txt", "w")
    for message in sentiment_scores:
        print("{0}: {1}".format(message[0], message[1].encode("utf-8")), file = output)
    
    output.close()


def print_conversation(i, conversations_dict):
    for message in conversations_dict[i]:
        print(message.encode("utf-8"))


def conversation_sentiment_analysis(classifier):
    print("Getting conversation sentiment scores...")
    
    conversations = open("Conversations.txt")
    
    old_message_time = None
    conversations_dict = {0: []}
    conversation = 0
    conversation_gap = 60 * 60
    pattern = "%Y-%m-%d %H:%M:%S"
    first_time = None
    senders = set()
    
    for line in conversations:
        line = line.strip().decode("utf-8")
        contents = line.split("[SEP]")
        if len(contents) < 3:
            continue
        
        timestamp = contents[0]
        if not first_time:
            first_time = int(time.mktime(time.strptime(timestamp, pattern)))
        sender = contents[1].replace(",", "")
        senders.add(sender)
        cur_message_time = int(time.mktime(time.strptime(timestamp, pattern)))
        if not old_message_time:
            old_message_time = cur_message_time
        diff = cur_message_time - old_message_time
        if diff > conversation_gap:
            conversation += 1
            conversations_dict[conversation] = []
        conversations_dict[conversation].append(line)
        old_message_time = cur_message_time
    
    sentiment_scores = []
    
    fieldnames = ["time", "overall"] + list(senders)
    writer = csv.DictWriter(open("Files/conversation_sentiments.csv", "w"), fieldnames)
    writer.writeheader()
    
    for i in range(0, len(conversations_dict)):
        conversation = conversations_dict[i]
        
        all_tokens = []
        sender_tokens = {sender: [] for sender in senders}
        
        time_diff = None
        
        for line in conversation:
            contents = line.split("[SEP]")
            sender = contents[1].replace(",", "")
            message = " ".join(contents[2:])
            timestamp = contents[0]
            if not time_diff:
                time_diff = int(time.mktime(time.strptime(timestamp, pattern))) - first_time
            tokens = nltk.word_tokenize(message)
            all_tokens += tokens
            sender_tokens[sender] += tokens
        
        all_feats = word_feats(all_tokens)
        sender_feats = {sender: word_feats(sender_tokens[sender]) for sender in senders}
        
        all_probs = classifier.prob_classify(all_feats)
        sender_probs = {sender: classifier.prob_classify(sender_feats[sender]) for sender in senders}
        
        all_pos = all_probs.prob("pos")
        sender_pos = {sender: sender_probs[sender].prob("pos") for sender in senders}
        
        sentiment_scores.append((all_pos, i))
        row = {"time": time_diff, "overall": all_pos}
        for sender in senders:
            row[sender] = sender_pos[sender]
        writer.writerow(row)
    
    sentiment_scores.sort(key = lambda conversation: conversation[0], reverse = True)
    
    output = open("Files/conversation_sentiment_scores.txt", "w")
    for conversation in sentiment_scores:
        separator = ["#" for i in range(0, 50)]
        separator[25] = str(conversation[1]) + "/" + str(conversation[0])
        print("".join(separator), file = output)
        for message in conversations_dict[conversation[1]]:
            print(message.encode("utf-8"), file = output)
    
    output.close()


def weekly_sentiment_analysis(classifier):
    print("Getting weekly sentiment scores...")
    
    conversations = open("Conversations.txt")
    
    senders = set()
    
    for line in conversations:
        line = line.strip().decode("utf-8")
        contents = line.split("[SEP]")
        if len(contents) < 3:
            continue
        
        sender = contents[1].replace(",", "")
        senders.add(sender)
    
    # Average weekly sentiment.
    fieldnames = ["time", "overall"] + list(senders)
    writer = csv.DictWriter(open("Files/weekly_sentiment.csv", "w"), fieldnames)
    writer.writeheader()
    
    conversations = open("Conversations.txt")
    
    pattern = "%Y-%m-%d %H:%M:%S"
    week_time = 60 * 60 * 24 * 7
    all_scores = []
    sender_scores = {sender: [] for sender in senders}
    
    start_time = None
    
    for line in conversations:
        line = line.strip().decode("utf-8")
        contents = line.split("[SEP]")
        if len(contents) < 3:
            continue
        
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        timestamp = contents[0]
        
        if not start_time:
            start_time = int(time.mktime(time.strptime(timestamp, pattern)))
        
        current_time = int(time.mktime(time.strptime(timestamp, pattern)))
        
        tokens = nltk.word_tokenize(message)
        feats = word_feats(tokens)
        probs = classifier.prob_classify(feats)
        pos = probs.prob("pos")
        
        if current_time - start_time > week_time:
            try:
                row = {"time": start_time, "overall": sum(all_scores) / len(all_scores)}
                for each_sender in senders:
                    row[each_sender] = sum(sender_scores[each_sender]) / len(sender_scores[each_sender])
                writer.writerow(row)
                start_time = current_time
                all_scores = [pos]
                for each_sender in senders:
                    if each_sender == sender:
                        sender_scores[each_sender] = [pos]
                    else:
                        sender_scores[each_sender] = []
            # At least one sender did not send a message during the current week, so we extend the time period.
            except ZeroDivisionError:
                all_scores.append(pos)
                sender_scores[sender].append(pos)
        else:
            all_scores.append(pos)
            sender_scores[sender].append(pos)
    
    # Last batch.
    row = {"time": start_time, "overall": sum(all_scores) / len(all_scores)}
    for each_sender in senders:
        row[each_sender] = sum(sender_scores[each_sender]) / len(sender_scores[each_sender])
    
    writer.writerow(row)


def main():
    run_sentiment_analysis()


if __name__ == "__main__":
    main()