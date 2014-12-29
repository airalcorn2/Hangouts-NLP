# Michael A. Alcorn (airalcorn2@gmail.com)

options(repos = structure(c(CRAN = "http://cran.revolutionanalytics.com/")))
install.packages(c("ggplot2", "reshape", "tm", "wordcloud"))

library(ggplot2)
library(reshape)
library(tm)
library(wordcloud)

# Message lengths.
data <- read.csv(file = "Files/messageLengths.csv", head = TRUE)
png("Files/messageLengths.png", width = 1000, height = 1000)
ggplot(data, aes(x = length, group = sender, fill = sender)) + stat_bin(aes(y = ..density..), binwidth = 1, position = 'dodge', alpha = 0.5) + coord_cartesian(xlim = c(0, 40))
dev.off()

# Message hours.
data <- read.csv(file = "Files/messageHours.csv", head = TRUE)
png("Files/messageHours.png", width = 1000, height = 1000)
ggplot(data, aes(x = hour, group = sender, fill = sender)) + stat_bin(aes(y = ..density..), binwidth = 1, position = 'dodge', alpha = 0.5)
dev.off()

# Message sentiment.
data <- read.csv(file = "Files/messageSentiments.csv", head = TRUE)
png("Files/messageSentiments.png", width = 1000, height = 1000)
ggplot(data, aes(x = sentiment, group = sender, fill = sender)) + stat_bin(aes(y = ..density..), binwidth = 0.01, position = 'identity', alpha = 0.5)
dev.off()

# Weekly sentiment.
data <- read.csv(file = "Files/weeklySentiment.csv", head = TRUE)
data$date <- as.Date(as.POSIXct(data$time, origin = "1970-01-01"))

melted <- melt(data, id = "date")
melted <- melted[melted$variable != "time", ]

png("Files/weeklySentiments.png", width = 1000, height = 1000)
ggplot(melted, aes(x = date, y = value, color = variable)) + geom_line()
dev.off()

# Word clouds.
messages <- Corpus(DirSource("Corpora"))
messages <- tm_map(messages, stripWhitespace)
messages <- tm_map(messages, content_transformer(tolower))
messages <- tm_map(messages, removePunctuation)
messages <- tm_map(messages, removeNumbers)
messages <- tm_map(messages, removeWords, stopwords("english"))

tdm <- TermDocumentMatrix(messages)
m <- as.matrix(tdm)

pdf("Files/comparisonCloud.pdf")
layout(matrix(c(1, 2), nrow = 2), heights = c(1, 10))
par(mar = rep(0, 4))
plot.new()
text(x = 0.5, y = 0.5, "Comparison Cloud")
comparison.cloud(m, max.words = 500, random.order = FALSE, main = "Title")
dev.off()

pdf("Files/commonalityCloud.pdf")
layout(matrix(c(1, 2), nrow = 2), heights = c(1, 10))
par(mar = rep(0, 4))
plot.new()
text(x = 0.5, y = 0.5, "Commonality Cloud")
commonality.cloud(m, max.words = 500, colors = brewer.pal(6, "Dark2"), random.order = FALSE)
dev.off()