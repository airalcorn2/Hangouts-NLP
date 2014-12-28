Hangouts-NLP
============

This program performs a number of natural language processing analyses on Google Hangouts instant messaging data.

<b>Requirements:</b>

<p>Linux</p>
<p>Bash (I get a UnicodeDecodeError when using Zsh for some reason)</p>
<p>Python (https://www.python.org/downloads/)</p>
<p>R (http://cran.revolutionanalytics.com/)</p>
<p>NLTK (http://www.nltk.org/install.html)</p>

<b>Step 1.</b>

Download your Google Hangouts data from https://www.google.com/settings/takeout.

<b>Step 2.</b>

Extract the compressed file.

<b>Step 3.</b>

Run:

<code>python hangouts.py '/path/to/Hangouts.json'</code>

<b>Step 4.</b>

Find the conversation id of the conversation you want to analyze.

<b>Step 5.</b>

Run:

<code>python hangouts.py '/path/to/Hangouts.json' -c &lt;conversation id&gt; > Conversations.txt</code>

<b>Step 6.</b>

Run:

<code>python runAnalysis.py</code>

<b>Step 7.</b>

Run:

<code>Rscript plots.R</code>
