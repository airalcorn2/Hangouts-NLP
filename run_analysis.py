#!/usr/bin/env python

from __future__ import print_function

import cluster_analysis
import os
import sender_analysis
import sentiment_analysis


def remove_blank_lines():
    conversations = open("RawConversations.txt")
    output = open("Conversations.txt", "w")
    current_line = conversations.readline().strip()
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        if len(contents) < 3:
            current_line += line.strip()
            continue
        else:
            print(current_line, file = output)
            current_line = line.strip()
    
    print(current_line, file = output)
    output.close()


def create_corpora():
    if not os.path.exists("Corpora"):
        os.makedirs("Corpora")
    
    conversations = open("Conversations.txt")
    
    sender_output = {}
    
    total_messages = 0
    
    for line in conversations:
        contents = line.strip().split("[SEP]")
        sender = contents[1].replace(",", "")
        message = " ".join(contents[2:])
        if sender not in sender_output:
            sender_output[sender] = open("Corpora/" + sender + "Messages", "w")
        print(message, file = sender_output[sender])
        total_messages += 1
    
    for sender in sender_output:
        sender_output[sender].close()

if __name__ == "__main__":
    if not os.path.exists("Files"):
        os.makedirs("Files")
    remove_blank_lines()
    sender_analysis.run_sender_analysis()
    sentiment_analysis.run_sentiment_analysis()
    cluster_analysis.run_cluster_analysis()
    create_corpora()