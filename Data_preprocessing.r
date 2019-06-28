library(jsonlite)
library(plyr)
library(tidyr)
library(data.table)
library(dplyr)
library(tidytext)
library(scales)
library(treemap)
library(tidytext)
library(text2vec)
library(glmnet)
library(janeaustenr)
library(igraph)
library(ggraph)
library(wordcloud)
library(knitr)
library(stringr)
library(text2vec)
library(plotrix)
fillColor = "#FFA07A"
fillColor2 = "#F1C40F"
fillColorBlue = "#AED6F1"
fillcolorblack="#000000"
fillcolorred="#FF0000"
fillcolorpurple = "#ACA4E2"
fillcoloryellow = "#F1C40F"
fillcolorblue = "#AED6F1"
fillcolororange="#FFA07A"
fillcolorpink="#E495A5"
fillcolorgreen="#39BEB1"
fillcolorpeach="#FB8072"

load_json <- function(filename) {
  js <- jsonlite::read_json(filename)
  data.table(
    id = sapply(js, `[[`, 'id'),
    ingredients = sapply(js, `[[`, 'ingredients'),
    cuisine = sapply(js, `[[`, 'cuisine')
  )
}

train3 <- load_json('train.json')
train3[, ingredients := lapply(ingredients, tolower)]
extract_text <- function() {
  mapply(function(ingredients, cuisine) {
    x <- paste(sample(ingredients), collapse=sprintf('#%s#', cuisine))
    x <- paste(cuisine, x, cuisine, sep='#')
    trimws(x)
  }, train3$ingredients, train3$cuisine)
}

set.seed(0)
texting <- do.call(c, replicate(100, extract_text(), simplify=F))
#Loading the Training Dataset
train <- fromJSON("train.json", flatten = TRUE)

#printing the No of Missing Value
print(paste("No of Missing Value in Training Dataset",sum(is.na(train))))


#Copying The data
train2 <- train

print(paste("No of rows present in Training Data",nrow(train)))
print(paste("No of Unique Cusines Present-",length(unique(train$cuisine))))

#Sorting the Ingredients in Training Data
train$ingredients <- lapply(train$ingredients, sort)

#converting the data to Lower case
train$ingredients <- lapply(train$ingredients,tolower)

#Remove the Duplicate Data
train<-train[!duplicated(train[c(2,3)]), ]

print(paste("No of rows present after removing the Duplicate cases in Training Data",nrow(train)))

#Every Ingredient Contain Only Unique element
train$ingredients<-lapply(train$ingredients,unique)



#Loading the Test Data
test <- fromJSON("test.json", flatten = TRUE)

#Sorting the ingredients of Test Data 
test$ingredients <- lapply(test$ingredients, sort)

#Converting to Lower form
test$ingredients <- lapply(test$ingredients, tolower)

#Printing the No of Missing Value in Test data
print(paste("No of Missing Value in Test Dataset",sum(is.na(test))))

#Fuction to convert the List of Ingredient in a Document form separated by #
ingredientscombine <- function(s)
{
  a <- unlist(s)
  return(paste0(a, collapse = '',sep='#'))
}

train$ingredients <- sapply(train$ingredients,ingredientscombine)

#Renaming Ingredients attribute into Text attribute 
train <- train %>%
  dplyr::rename(text = ingredients)

test$ingredients <- sapply(test$ingredients,ingredientscombine)
test <- test %>%
  dplyr::rename(text = ingredients)

#Counting No of Rows present
TotalNoofRows <- nrow(train)




cuisine_type = train %>%
  group_by(cuisine) %>%
  dplyr::summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(cuisine = reorder(cuisine,Count)) %>%
  head(20) 


treemap(cuisine_type, 
        index="cuisine", 
        vSize = "Count",  
        title="Cuisine Counts", 
        palette = "RdBu",
        fontsize.title = 14 
)
#Plotting the Graph of the Top cusines vs Percentage of it's Occurances    
train %>%
  group_by(cuisine) %>%
  dplyr::summarise(Count = n()/TotalNoofRows) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  mutate(cuisine = reorder(cuisine,Count)) %>%
  head(10) %>%
  ggplot(aes(x = cuisine,y = Count)) +
  geom_bar(stat='identity',fill= fillColor) +
  geom_text(aes(x = cuisine, y = .01, label = paste0("( ",round(Count*100,2)," %)",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  scale_y_continuous(labels = percent_format()) +
  labs(x = 'cuisine', 
       y = 'Percentage', 
       title = 'cuisine and Percentage') +
  coord_flip() +
  theme_bw()


 
train_count_by_id <-  train2 %>% 
  mutate(ingredients = str_split(ingredients, pattern = ",")) %>% 
  unnest(ingredients) %>% 
  mutate(ingredients = gsub(ingredients, pattern = 'c\\(', replacement = "")) %>%
  mutate(ingredients = gsub(ingredients, pattern = '"', replacement = "")) %>%
  mutate(ingredients = trimws(ingredients)) %>%
  group_by(id,cuisine) %>%
  dplyr::summarise(CountOfIngredients = n())


train_count_by_id %>%
  group_by(cuisine) %>%
  dplyr::summarise(MedianCountOfIngredients = median(CountOfIngredients,na.rm=TRUE)) %>%
  arrange(desc(MedianCountOfIngredients)) %>%
  ungroup() %>%
  mutate(cuisine = reorder(cuisine,MedianCountOfIngredients)) %>%
  head(10) %>%
  
  ggplot(aes(x = cuisine,y = MedianCountOfIngredients)) +
  geom_bar(stat='identity',fill= fillColorBlue) +
  geom_text(aes(x = cuisine, y = .01, label = paste0("( ",round(MedianCountOfIngredients,2)," )",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'cuisine', 
       y = 'MedianCountOfIngredients', 
       title = 'cuisine and MedianCountOfIngredients') +
  coord_flip() +
  theme_bw()
train_count_by_id %>%
  group_by(cuisine) %>%
  dplyr::summarise(MedianCountOfIngredients = median(CountOfIngredients,na.rm=TRUE)) %>%
  arrange(desc(MedianCountOfIngredients)) %>%
  ungroup() %>%
  mutate(cuisine = reorder(cuisine,MedianCountOfIngredients)) %>%
  tail(10) %>%
  
  ggplot(aes(x = cuisine,y = MedianCountOfIngredients)) +
  geom_bar(stat='identity',fill= fillColor2) +
  geom_text(aes(x = cuisine, y = .01, label = paste0("( ",round(MedianCountOfIngredients,2)," )",sep="")),
            hjust=0, vjust=.5, size = 4, colour = 'black',
            fontface = 'bold') +
  labs(x = 'cuisine', 
       y = 'MedianCountOfIngredients', 
       title = 'cuisine and MedianCountOfIngredients') +
  coord_flip() +
  theme_bw()
 

createBarPlotCommonWords = function(train,titleName)
{
  train %>% 
    mutate(ingredients = str_split(ingredients, pattern = ",")) %>% 
    unnest(ingredients) %>% 
    mutate(ingredients = gsub(ingredients, pattern = 'c\\(', replacement = "")) %>%
    mutate(ingredients = gsub(ingredients, pattern = '"', replacement = "")) %>%
    mutate(ingredients = trimws(ingredients)) %>%
    group_by(ingredients) %>%
    dplyr::summarise(Count = n()) %>%
    arrange(desc(Count)) %>%
    ungroup() %>%
    mutate(ingredients = reorder(ingredients,Count)) %>%
    head(10) %>%
    
    
    ggplot(aes(x = ingredients,y = Count)) +
    geom_bar(stat='identity',fill= fillColor) +
    geom_text(aes(x = ingredients, y = .01, label = paste0("( ",Count," )",sep="")),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
    labs(x = 'ingredients', 
         y = 'Count', 
         title = titleName) +
    coord_flip() +
    theme_bw()
  
}

createBarPlotCommonWords(train2,'Top 10 most Common Ingredients')
most_common_ingredients <- train2 %>% 
  mutate(ingredients = str_split(ingredients, pattern = ",")) %>% 
  unnest(ingredients) %>% 
  mutate(ingredients = gsub(ingredients, pattern = 'c\\(', replacement = "")) %>%
  mutate(ingredients = gsub(ingredients, pattern = '"', replacement = "")) %>%
  mutate(ingredients = trimws(ingredients)) %>%
  group_by(ingredients) %>%
  dplyr::summarise(Count = n()) %>%
  arrange(desc(Count)) %>%
  ungroup() %>%
  head(10)

createBarPlotCommonWordsInCuisine = function(train,cuisineName,titleName,fillColorName)
{
  train %>% 
    filter(cuisine == cuisineName) %>%
    mutate(ingredients = str_split(ingredients, pattern = ",")) %>% 
    unnest(ingredients) %>% 
    mutate(ingredients = gsub(ingredients, pattern = 'c\\(', replacement = "")) %>%
    mutate(ingredients = gsub(ingredients, pattern = '"', replacement = "")) %>%
    mutate(ingredients = trimws(ingredients)) %>%
    filter(!ingredients %in% most_common_ingredients$ingredients) %>%
    group_by(ingredients) %>%
    dplyr::summarise(Count = n()) %>%
    arrange(desc(Count)) %>%
    ungroup() %>%
    mutate(ingredients = reorder(ingredients,Count)) %>%
    head(10) %>%
    ggplot(aes(x = ingredients,y = Count)) +
    geom_bar(stat='identity',fill= fillColorName) +
    geom_text(aes(x = ingredients, y = .01, label = paste0("",sep="")),
              hjust=0, vjust=.5, size = 4, colour = 'black',
              fontface = 'bold') +
    labs(x = 'ingredients', 
         y = 'Count', 
         title = titleName) +
    coord_flip() +
    theme_bw()
  
}

#Plotted the graph of most common Ingredients present in cusine
createBarPlotCommonWordsInCuisine(train2,"italian","Most Common Ingredients in Italian Cuisine",fillcolorred)+ coord_polar("y", start=0)
createBarPlotCommonWordsInCuisine(train2,"mexican","Most Common Ingredients in Mexican Cuisine",fillcolorblue)+ coord_polar("y", start=0)
createBarPlotCommonWordsInCuisine(train2,"thai","Most Common Ingredients in Thai Cuisine",fillcolorpeach)+ coord_polar("y", start=0)
createBarPlotCommonWordsInCuisine(train2,"moroccan","Most Common Ingredients in Moroccan Cuisine",fillcolorgreen)+ coord_polar("y", start=0)


plotMostImportantWords <- function(train) {
  trainWords <- train %>%
    unnest_tokens(word, text) %>%
    dplyr::count(cuisine, word, sort = TRUE) %>%
    ungroup()
  
  total_words <- trainWords %>% 
    group_by(cuisine) %>% 
    dplyr::summarize(total = sum(n))
  
  trainWords <- left_join(trainWords, total_words)
  
  #Now we are ready to use the bind_tf_idf which computes the tf-idf for each term. 
  trainWords <- trainWords %>%
    filter(!is.na(cuisine)) %>%
    bind_tf_idf(word, cuisine, n)
  
  
  plot_trainWords <- trainWords %>%
    arrange(desc(tf_idf)) %>%
    mutate(word = factor(word, levels = rev(unique(word))))
  
  return(plot_trainWords)
}

plot_trainWords <- plotMostImportantWords(train)

plot_trainWords %>% 
  top_n(20) %>%
  ggplot(aes(word, tf_idf)) +
  geom_col(fill = fillColor2) +
  labs(x = NULL, y = "tf-idf") +
  coord_flip() +
  theme_bw()

plotMostImportantIngredientsInCuisine <- function(plot_trainWords, cuisineName,fillColorName = fillColor) {
  plot_trainWords %>% 
    filter(cuisine == cuisineName) %>%
    top_n(10) %>%
    ggplot(aes(word, tf_idf)) +
    geom_col(fill = fillColorName) +
    labs(x = NULL, y = "tf-idf") +
    coord_flip() +
    theme_bw()
}

plotMostImportantIngredientsInCuisine(plot_trainWords,"italian",fillcolorblack)+ coord_polar("y", start=0)
plotMostImportantIngredientsInCuisine(plot_trainWords,"mexican",fillColor2)+ coord_polar("y", start=0)
plotMostImportantIngredientsInCuisine(plot_trainWords,"thai",fillcolorpink)+ coord_polar("y", start=0)
plotMostImportantIngredientsInCuisine(plot_trainWords,"moroccan",fillcolororange)+ coord_polar("y", start=0)
# create glove vectors

special_tokenizer <- function(x, ...) space_tokenizer(x, sep = "#", ...)
it <- itoken(texting, tokenizer = special_tokenizer)

# Glove Word Embedding
# Create vocabulary, terms will be unigrams
vocab <- create_vocabulary(it)
vectorizer <- vocab_vectorizer(vocab)

# use window of 4 for context words
tcm <- create_tcm(it, vectorizer, skip_grams_window=4L)

# create glove vectors
glove <- GlobalVectors$new(word_vectors_size=100, vocabulary=vocab, x_max=100)
invisible(capture.output(wv_matrix <- glove$fit_transform(tcm, n_iter=500)))

cuisine_list <- train3[, unique(cuisine)]
cuisine_matrix <- wv_matrix[cuisine_list,]

# normalize vectors
normalize_l2 <- function(x) {x / sqrt(sum(x^2))}
cuisine_matrix <- t(apply(cuisine_matrix, 1, normalize_l2))
head(cuisine_matrix, 3)

similarity_matrix <- cuisine_matrix %*% t(cuisine_matrix)
similarity_matrix <- matrix(ecdf(similarity_matrix)(similarity_matrix),
                            nrow(similarity_matrix), ncol(similarity_matrix))
colnames(similarity_matrix) <- rownames(similarity_matrix) <- cuisine_list

options(repr.plot.width=8, repr.plot.height=8)
pallete=colorRampPalette(c("white","yellow","red","black"),space="rgb") 
heatmap(similarity_matrix, symm=TRUE,col=pallete(20))


library(Rtsne)
tsne_plot <- function(tsne_matrix, ...) {
  options(repr.plot.width=8, repr.plot.height=6)
  x <- tsne_matrix[,1]
  y <- tsne_matrix[,2]
  plot(x, y, pch=16, col='blue', cex=0.5, xlim=1.2*range(x), ylim=1.2*range(y))
  text(x, y, pos=1, ...) 
}
set.seed(1)
cuisine_tsne <- Rtsne(cuisine_matrix, perplexity=2.25, max_iter=1e5)$Y
tsne_plot(cuisine_tsne, labels=rownames(cuisine_matrix))

#Replacing the " "  by "_"
train$text <- gsub(" ","_",train$text)
#Replacing the "#" by " "
train$text <- gsub("#"," ",train$text) 

#prepreocessing function for Converting into DTM
prep_fun  = function(x) {
  stringr::str_replace_all(tolower(x), "[^[:alpha:]]", " ")
}

  #Taking the word Tokenizer 
tok_fun = word_tokenizer

#Creating a object to Create a vocabulary for training Data
it_train = itoken(train$text,
                  preprocessor = prep_fun,
                  tokenizer = tok_fun,
                  ids = train$id,
                  progressbar = FALSE)

#Creating the object to create vocabulary for test data
it_test = test$text %>% 
  prep_fun %>% 
    tok_fun %>% 
  itoken(ids = test$id,  progressbar = FALSE)


NFOLDS = 4

#Generated the vocab for whole datset
vocab = create_vocabulary(it_train, ngram = c(1L,2L))

#vocabolary before Trimming
vocab

#Vocabulary after trimming the Ingredients having more than 50% occurance as well as less than 1% occurance
vocab = vocab %>% prune_vocabulary(term_count_min = 10, 
                                   doc_proportion_max = 0.5,
                                   doc_proportion_min = 0.01,vocab_term_max = 5000)

vocab

trigram_vectorizer = vocab_vectorizer(vocab)

#Generate the Document Term matrix using trimmed Vocab
dtm_train = create_dtm(it_train, trigram_vectorizer)
Train_dtm<-as.data.frame(as.matrix(dtm_train))
Train_dtm<-data.frame(Train_dtm[0:0],cuisine=train$cuisine,Train_dtm[2:ncol(Train_dtm)])
#Dimension of Finally Obtanined Training_DTM
dim(Train_dtm)
tfidf = TfIdf$new(norm = "l2", sublinear_tf = T)

# fit model to train data and transform train data with fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)
Train_dtm_tfidf<-as.data.frame(as.matrix(dtm_train_tfidf))

# tfidf modified by fit_transform() call!
# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf = create_dtm(it_test, trigram_vectorizer)

dtm_test_tfidf = transform(dtm_test_tfidf, tfidf)

Test_dtm_tfidf<-as.data.frame(as.matrix(dtm_test_tfidf))


