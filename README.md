Hangouts-NLP
============

This program performs a number of different natural language processing analyses on Google Hangouts instant messaging data.

<b>Requirements:</b>

<p>Linux</p>
<p>Bash (I get a UnicodeDecodeError when using Zsh for some reason)</p>
<p>Python (https://www.python.org/downloads/)</p>
<p>R (http://cran.revolutionanalytics.com/)</p>
<p>NLTK (http://www.nltk.org/install.html)</p>

<b>Step 1.</b>

Download your Google Hangouts data from https://www.google.com/settings/takeout.

<b>Step 2.</b>

Extract the compressed file and place it in the directory containing the code.

<b>Step 3.</b>

Run:

<code>python hangouts-log-reader/hangouts.py Hangouts.json</code>

<b>Step 4.</b>

Find the conversation id of the conversation you want to analyze.

<b>Step 5.</b>

Run:

<code>python hangouts-log-reader/hangouts.py Hangouts.json -c &lt;conversation id&gt; > Conversations.txt</code>

<b>Step 6.</b>

Download the sentiment training file found here -> https://drive.google.com/file/d/0B4iRo-F4K4f8VXFhd2pLWUJTdTg/view?usp=sharing and place it in the directory containing the code. See http://help.sentiment140.com/for-students/ for details.

<b>Step 7.</b>

Run:

<code>python runAnalysis.py</code>

<b>Step 8.</b>

Run:

<code>Rscript plots.R</code>
