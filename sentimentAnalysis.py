from __future__ import print_function

import csv
import nltk
import sys
import time

from nltk.classify import NaiveBayesClassifier

def word_feats(words):
    return dict([(word, True) for word in words])

def runSentimentAnalysis(checkContiguousMessages = True, checkConversations = True, checkWeeks = True):
    
    print("Performing sentiment analysis...")
    
    f = open("sentimentTraining")
    tweets = [line.strip() for line in f]
    totalTweets = len(tweets)
    
    features = []
    
    count = 0
    
    # Takes a while.
    for tweet in tweets:
        if count % 10000 == 0:
            percent = 100 * count / totalTweets
            sys.stdout.write("\rTraining {0}% complete...".format(percent))
            sys.stdout.flush()
        [sentiment, sentence] = tweet.split("[SEP]")
        # sentence = unicode(sentence, 'utf-8')
        posOrNeg = None
        if sentiment == "0":
            posOrNeg = "neg"
        elif sentiment == "4":
            posOrNeg = "pos"
        else:
            continue
        tokens = nltk.word_tokenize(sentence)
        feats = word_feats(tokens)
        features.append((feats, posOrNeg))
        count += 1
    
    print()
    classifier = NaiveBayesClassifier.train(features)
    
    classifier.show_most_informative_features(100)
    
    messageSentimentAnalysis(classifier)
    if checkContiguousMessages:
        contiguousMessagesSentimentAnalysis(classifier)
    
    if checkConversations:
        conversationSentimentAnalysis(classifier)
    
    if checkWeeks:
        weeklySentimentAnalysis(classifier)

def messageSentimentAnalysis(classifier):
    print("Getting message sentiment scores...")
    
    conversations = open("Conversations.txt")
    
    fieldNames = ["sentiment", "sender"]
    writer = csv.DictWriter(open("Files/messageSentiments.csv", "w"), fieldNames)
    writer.writeheader()
    
    sentimentScores = []
    
    # Individual messages.
    for line in conversations:
        contents = line.strip().split("[SEP]")
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        tokens = nltk.word_tokenize(message)
        feats = word_feats(tokens)
        probs = classifier.prob_classify(feats)
        pos = probs.prob("pos")
        row = {"sentiment": pos, "sender": sender}
        writer.writerow(row)
        sentimentScores.append((pos, message))
    
    sentimentScores.sort(key = lambda message: message[0], reverse = True)
    
    output = open("Files/messageSentimentScores", "w")
    for message in sentimentScores:
        print("{0}: {1}".format(message[0], message[1]), file = output)
    
    output.close()

def contiguousMessagesSentimentAnalysis(classifier):
    print("Getting contiguous messages sentiment scores...")
    
    conversations = open("Conversations.txt")
    
    fieldNames = ["sentiment", "sender"]
    writer = csv.DictWriter(open("Files/contiguousMessageSentiments.csv", "w"), fieldNames)
    writer.writeheader()
    
    sentimentScores = []
    currentSender = None
    currentMessage = ""
    currentTokens = []
    pattern = "%Y-%m-%d %H:%M:%S"
    gapTime = 60 * 10
    prevTime = None
    
    # Contiguous messages.
    for line in conversations:
        contents = line.strip().split("[SEP]")
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        timestamp = contents[0]
        if not currentSender:
            currentSender = sender
            prevTime = int(time.mktime(time.strptime(timestamp, pattern)))
        tokens = nltk.word_tokenize(message)
        currentTime = int(time.mktime(time.strptime(timestamp, pattern)))
        if sender == currentSender and currentTime - prevTime <= gapTime:
            currentTokens += tokens
            currentMessage += message + " "
            prevTime = currentTime
            continue
        else:
            feats = word_feats(currentTokens)
            probs = classifier.prob_classify(feats)
            pos = probs.prob("pos")
            row = {"sentiment": pos, "sender": sender}
            writer.writerow(row)
            sentimentScores.append((pos, currentMessage[:-1]))
            currentSender = sender
            prevTime = currentTime
            currentMessage = message + " "
            currentTokens = tokens
    
    # Last batch.
    feats = word_feats(currentTokens)
    probs = classifier.prob_classify(feats)
    pos = probs.prob("pos")
    row = {"sentiment": pos, "sender": sender}
    writer.writerow(row)
    
    sentimentScores.append((pos, currentMessage[:-1]))
    
    sentimentScores.sort(key = lambda message: message[0], reverse = True)
    
    output = open("Files/contiguousMessageSentimentScores", "w")
    for message in sentimentScores:
        print("{0}: {1}".format(message[0], message[1]), file = output)
    
    output.close()

def printConversation(i, conversationsDict):
    for message in conversationsDict[i]:
        print(message)

def conversationSentimentAnalysis(classifier):
    print("Getting conversation sentiment scores...")
    
    conversations = open("Conversations.txt")
    
    oldMessageTime = None
    conversationsDict = {0: []}
    conversation = 0
    conversationGap = 60 * 60
    pattern = "%Y-%m-%d %H:%M:%S"
    senders = {}
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        timestamp = contents[0]
        sender = contents[1].replace(",", "")
        if sender not in senders:
            senders[sender] = True
        curMessageTime = int(time.mktime(time.strptime(timestamp, pattern)))
        if not oldMessageTime:
            oldMessageTime = curMessageTime
        diff = curMessageTime - oldMessageTime
        if diff > conversationGap:
            conversation += 1
            conversationsDict[conversation] = []
        conversationsDict[conversation].append(line)
        oldMessageTime = curMessageTime
    
    sentimentScores = []
    
    fieldNames = ["time", "overall"] + senders.keys()
    
    writer = csv.DictWriter(open("Files/conversationSentiments.csv", "w"), fieldNames)
    writer.writeheader()
    
    for i in range(0, len(conversationsDict.keys())):
        conversation = conversationsDict[i]
        
        allTokens = []
        senderTokens = {}
        for sender in senders.keys():
            senderTokens[sender] = []
        timeDiff = None
        
        for line in conversation:
            contents = line.split("[SEP]")
            sender = contents[1].replace(",", "")
            message = " ".join(contents[2:])
            timestamp = contents[0]
            if not timeDiff:
                timeDiff = int(time.mktime(time.strptime(timestamp, pattern))) - firstTime
            tokens = nltk.word_tokenize(message)
            allTokens += tokens
            senderTokens[sender] += tokens
        
        allFeats = word_feats(allTokens)
        senderFeats = {}
        for sender in senders.keys():
            senderFeats[sender] = word_feats(senderTokens[sender])
        
        allProbs = classifier.prob_classify(allFeats)
        senderProbs = {}
        for sender in senders.keys():
            senderProbs[sender] = classifier.prob_classify(senderFeats[sender])
        
        allPos = allProbs.prob("pos")
        senderPos = {}
        for sender in senders.keys():
            senderPos[sender] = senderProbs[sender].prob("pos")
        
        sentimentScores.append((allPos, i))
        row = {"time": timeDiff, "overall": allPos}
        for sender in senders.keys():
            row[sender] = senderPos[sender]
        writer.writerow(row)
    
    sentimentScores.sort(key = lambda conversation: conversation[0], reverse = True)
    
    output = open("Files/conversationSentimentScores", "w")
    for conversation in sentimentScores:
        separator = ["#" for i in range(0, 50)]
        separator[25] = str(conversation[1]) + "/" + str(conversation[0])
        print("".join(separator), file = output)
        for message in conversationsDict[conversation[1]]:
            print(message, file = output)
    
    output.close()

def weeklySentimentAnalysis(classifier):
    print("Getting weekly sentiment scores...")
    
    conversations = open("Conversations.txt")
    
    senders = {}
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        sender = contents[1].replace(",", "")
        if sender not in senders:
            senders[sender] = True
    
    # Average weekly sentiment.
    fieldNames = ["time", "overall"] + senders.keys()
    writer = csv.DictWriter(open("Files/weeklySentiment.csv", "w"), fieldNames)
    writer.writeheader()
    
    conversations = open("Conversations.txt")
    
    pattern = "%Y-%m-%d %H:%M:%S"
    weekTime = 60 * 60 * 24 * 7
    allScores = []
    senderScores = {}
    for sender in senders.keys():
        senderScores[sender] = []
    
    startTime = None
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        timestamp = contents[0]
        
        if not startTime:
            startTime = int(time.mktime(time.strptime(timestamp, pattern)))
        
        currentTime = int(time.mktime(time.strptime(timestamp, pattern)))
        
        tokens = nltk.word_tokenize(message)
        feats = word_feats(tokens)
        probs = classifier.prob_classify(feats)
        pos = probs.prob("pos")
        
        if currentTime - startTime > weekTime:
            try:
                row = {"time": startTime, "overall": sum(allScores) / len(allScores)}
                for eachSender in senders.keys():
                    row[eachSender] = sum(senderScores[eachSender]) / len(senderScores[eachSender])
                writer.writerow(row)
                startTime = currentTime
                allScores = [pos]
                for eachSender in senders.keys():
                    if eachSender == sender:
                        senderScores[eachSender] = [pos]
                    else:
                        senderScores[eachSender] = []
            # At least one sender did not send a message during the current week, so we extend the time period.
            except ZeroDivisionError:
                allScores.append(pos)
                senderScores[sender].append(pos)
        else:
            allScores.append(pos)
            senderScores[sender].append(pos)
    
    # Last batch.
    row = {"time": startTime, "overall": sum(allScores) / len(allScores)}
    for eachSender in senders.keys():
        row[eachSender] = sum(senderScores[eachSender]) / len(senderScores[eachSender])
    
    writer.writerow(row)


if __name__ == "__main__":
    runSentimentAnalysis()