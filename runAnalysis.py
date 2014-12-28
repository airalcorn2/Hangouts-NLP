from __future__ import print_function

import clusterAnalysis
import senderAnalysis
import sentimentAnalysis

def createCorpora():
    conversations = open("Conversations.txt")
    
    line = conversations.readline().strip()
    allOutput = open("allMessages", "w")
    michaelOutput = open("michaelMessages", "w")
    katherineOutput = open("katherineMessages", "w")
    totalMessages = 0
    
    while line != "":
        [timestamp, sender, message] = line.split("[SEP]")
        if "Michael" in sender:
            print(message, file = michaelOutput)
        else:
            print(message, file = katherineOutput)
        print(message, file = allOutput)
        line = conversations.readline().strip()
        totalMessages += 1
    
    allOutput.close()
    michaelOutput.close()
    katherineOutput.close()
    
    conversations = open("Conversations.txt")
    
    line = conversations.readline().strip()
    allMessages = []
    
    while line != "":
        allMessages.append(line)
        line = conversations.readline().strip()
    
    groupSize = (len(allMessages) / 5) + 1
    phases = []
    for i in range(0, 5):
        start = i * groupSize
        end = start + groupSize
        phases.append(allMessages[start:end])
    
    phaseCount = 1
    for phase in phases:
        allOutput = open("allMessagesPhase{0}".format(phaseCount), "w")
        michaelOutput = open("michaelMessagesPhase{0}".format(phaseCount), "w")
        katherineOutput = open("katherineMessagesPhase{0}".format(phaseCount), "w")
        for line in phase:
            [timestamp, sender, message] = line.split("[SEP]")
            if "Michael" in sender:
                print(message, file = michaelOutput)
            else:
                print(message, file = katherineOutput)
            print(message, file = allOutput)
            line = conversations.readline().strip()
        allOutput.close()
        michaelOutput.close()
        katherineOutput.close()
        phaseCount += 1

if __name__ == "__main__":
    senderAnalysis.runSenderAnalysis()
    sentimentAnalysis.runSentimentAnalysis()
    clusterAnalysis.runClusterAnalysis()