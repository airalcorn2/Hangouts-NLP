from __future__ import print_function

import clusterAnalysis
import senderAnalysis
import sentimentAnalysis

def createCorpora():
    conversations = open("Conversations.txt")
    
    line = conversations.readline().strip()
    allOutput = open("allMessages", "w")
    senderOutput = {}
    
    totalMessages = 0
    
    while line != "":
        contents = line.split("[SEP]")
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        if sender not in senderOutput:
            senderOutput[sender] = open(sender + "Messages", "w")
        print(message, file = senderOutput[sender])
        print(message, file = allOutput)
        line = conversations.readline().strip()
        totalMessages += 1
    
    allOutput.close()
    for sender in senderOutput.keys():
        senderOutput[sender].close()
    
    conversations = open("Conversations.txt")
    
    line = conversations.readline().strip()
    allMessages = []
    
    while line != "":
        allMessages.append(line)
        line = conversations.readline().strip()
    
    numberOfPhases = 5
    
    groupSize = (len(allMessages) / numberOfPhases) + 1
    phases = []
    for i in range(0, numberOfPhases):
        start = i * groupSize
        end = start + groupSize
        phases.append(allMessages[start:end])
    
    phaseCount = 1
    for phase in phases:
        allOutput = open("allMessagesPhase{0}".format(phaseCount), "w")
        senderOutput = {}
        for line in phase:
            contents = line.split("[SEP]")
            sender = contents[1].replace(",", "")
            message = " ".join(contents[2:])
            if sender not in senderOutput:
                senderOutput[sender] = open(sender + "MessagesPhase{0}".format(phaseCount), "w")
            print(message, file = senderOutput[sender])
            print(message, file = allOutput)
        allOutput.close()
        for sender in senderOutput.keys():
            senderOutput[sender].close()
        phaseCount += 1

if __name__ == "__main__":
    senderAnalysis.runSenderAnalysis()
    sentimentAnalysis.runSentimentAnalysis()
    clusterAnalysis.runClusterAnalysis()
    createCorpora()