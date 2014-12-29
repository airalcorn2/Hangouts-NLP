from __future__ import print_function

import clusterAnalysis
import os
import senderAnalysis
import sentimentAnalysis

def createCorpora():
    if not os.path.exists("Corpora"):
        os.makedirs("Corpora")
    
    conversations = open("Conversations.txt")
    
    line = conversations.readline().strip()
    senderOutput = {}
    
    totalMessages = 0
    
    while line != "":
        contents = line.split("[SEP]")
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        if sender not in senderOutput:
            senderOutput[sender] = open("Corpora/" + sender + "Messages", "w")
        print(message, file = senderOutput[sender])
        line = conversations.readline().strip()
        totalMessages += 1
    
    for sender in senderOutput.keys():
        senderOutput[sender].close()

if __name__ == "__main__":
    if not os.path.exists("Files"):
        os.makedirs("Files")
    senderAnalysis.runSenderAnalysis()
    sentimentAnalysis.runSentimentAnalysis()
    clusterAnalysis.runClusterAnalysis()
    createCorpora()