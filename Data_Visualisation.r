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
