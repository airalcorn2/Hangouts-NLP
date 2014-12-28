# Michael A. Alcorn (airalcorn2@gmail.com)

library(ggplot2)
library(reshape)

library(plyr)

library(tm)
library(wordcloud)

# Message lengths.
data <- read.csv(file = "messageLengths.csv", head = TRUE)
png("messageLengths.png", width = 1000, height = 1000)
ggplot(data, aes(x = length, group = sender, fill = sender)) + stat_bin(aes(y = ..density..), binwidth = 1, position = 'dodge', alpha = 0.5) + coord_cartesian(xlim = c(0, 40))
dev.off()

# Message hours.
data <- read.csv(file = "messageHours.csv", head = TRUE)
png("messageHours.png", width = 1000, height = 1000)
ggplot(data, aes(x = hour, group = sender, fill = sender)) + stat_bin(aes(y = ..density..), binwidth = 1, position = 'dodge', alpha = 0.5)
dev.off()

# Message sentiment.
data <- read.csv(file = "messageSentiments.csv", head = TRUE)
png("messageSentiments.png", width = 1000, height = 1000)
ggplot(data, aes(x = sentiment, group = sender, fill = sender)) + stat_bin(aes(y = ..density..), binwidth = 0.01, position = 'identity', alpha = 0.5)
dev.off()

# Weekly sentiment.
data <- read.csv(file = "weeklySentiment.csv", head = TRUE)
data$date <- as.Date(as.POSIXct(data$time, origin = "1970-01-01"))

melted <- melt(data, id = "date")
melted <- melted[melted$variable != "time", ]

png("weeklySentiments.png", width = 1000, height = 1000)
ggplot(melted, aes(x = date, y = value, color = variable)) + geom_line()
dev.off()

# Word clouds.
messages <- Corpus(DirSource("Corpora/Split/"))
messages <- tm_map(messages, stripWhitespace)
messages <- tm_map(messages, content_transformer(tolower))
messages <- tm_map(messages, removePunctuation)
messages <- tm_map(messages, removeNumbers)
messages <- tm_map(messages, removeWords, stopwords("english"))
wordcloud(messages, colors = brewer.pal(6, "Dark2"), random.order = FALSE)

tdm <- TermDocumentMatrix(messages)
m <- as.matrix(tdm)

colnames(m) <- c("Katherine", "Michael")
layout(matrix(c(1, 2), nrow = 2), heights = c(1, 10))
par(mar = rep(0, 4))
plot.new()
text(x = 0.5, y = 0.5, "Comparison Cloud")
comparison.cloud(m, max.words = 500, random.order = FALSE, main = "Title")

commonality.cloud(m, max.words = 500, colors = brewer.pal(6, "Dark2"), random.order = FALSE)

v <- sort(rowSums(m), decreasing = TRUE)
d <- data.frame(word = names(v), freq = v)
quantile(d$freq, c(0.25, 0.5, 0.75, 0.9, 0.95, 0.99))
cut350 <- d[d$freq < 350, ]
cut150 <- d[d$freq < 150, ]

wordcloud(d$word, d$freq, colors = brewer.pal(6, "Dark2"), random.order = FALSE)
wordcloud(cut350$word, cut350$freq, colors = brewer.pal(6, "Dark2"), random.order = FALSE)
wordcloud(cut150$word, cut150$freq, colors = brewer.pal(6, "Dark2"), random.order = FALSE)

fileName <- "allMessages"
text <- readChar(fileName, file.info(fileName)$size)
wordcloud(text, random.order = FALSE)

directory <- c("All", "Phase1", "Phase2", "Phase3", "Phase4", "Phase5")
for (i in seq(1, 6))
{

    dir <- paste("Corpora/Split/", directory[i], "/", sep = "")

    messages <- Corpus(DirSource(dir))
    messages <- tm_map(messages, stripWhitespace)
    messages <- tm_map(messages, content_transformer(tolower))
    messages <- tm_map(messages, removePunctuation)
    messages <- tm_map(messages, removeNumbers)
    messages <- tm_map(messages, removeWords, stopwords("english"))

    pdf(paste("wordcloud", directory[i], ".pdf", sep = ""))
    wordcloud(messages, colors = brewer.pal(6, "Dark2"), random.order = FALSE)
    dev.off()

    tdm <- TermDocumentMatrix(messages)
    m <- as.matrix(tdm)
    colnames(m) <- c("Katherine", "Michael")

    pdf(paste("comparisoncloud", directory[i], ".pdf", sep = ""))
    layout(matrix(c(1, 2), nrow = 2), heights = c(1, 10))
    par(mar = rep(0, 4))
    plot.new()
    figTitle <- paste("A comparison word cloud for Phase ", i - 1, " of the relationship.", sep = "")
    text(x = 0.5, y = 0.5, figTitle)
    comparison.cloud(m, max.words = 500, random.order = FALSE, main = "Title")
    dev.off()

    pdf(paste("commonalitycloud", directory[i], ".pdf", sep = ""))
    layout(matrix(c(1, 2), nrow = 2), heights = c(1, 10))
    par(mar = rep(0, 4))
    plot.new()
    figTitle <- paste("A commonality word cloud for Phase ", i - 1, " of the relationship.", sep = "")
    text(x = 0.5, y = 0.5, figTitle)
    commonality.cloud(m, max.words = 500, colors = brewer.pal(6, "Dark2"), random.order = FALSE, main = "Title")
    dev.off()
}
