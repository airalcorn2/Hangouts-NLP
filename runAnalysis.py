from __future__ import print_function

import clusterAnalysis
import os
import senderAnalysis
import sentimentAnalysis

def removeBlankLines():
    conversations = open("RawConversations.txt")
    output = open("Conversations.txt", "w")
    currentLine = conversations.readline()
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        if len(contents) < 3:
            currentLine += line.strip()
            continue
        else:
            print(currentLine, file = output)
            currentLine = line.strip()
    
    print(currentLine, file = output)
    output.close()


def createCorpora():
    if not os.path.exists("Corpora"):
        os.makedirs("Corpora")
    
    conversations = open("Conversations.txt")
    
    senderOutput = {}
    
    totalMessages = 0
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        if sender not in senderOutput:
            senderOutput[sender] = open("Corpora/" + sender + "Messages", "w")
        print(message, file = senderOutput[sender])
        totalMessages += 1
    
    for sender in senderOutput.keys():
        senderOutput[sender].close()

if __name__ == "__main__":
    if not os.path.exists("Files"):
        os.makedirs("Files")
    removeBlankLines()
    senderAnalysis.runSenderAnalysis()
    sentimentAnalysis.runSentimentAnalysis()
    clusterAnalysis.runClusterAnalysis()
    createCorpora()