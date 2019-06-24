library(data.table)
library(text2vec)
load_json <- function(filename) {
  js <- jsonlite::read_json(filename)
  data.table(
    id = sapply(js, `[[`, 'id'),
    ingredients = sapply(js, `[[`, 'ingredients'),
    cuisine = sapply(js, `[[`, 'cuisine')
  )
}
train <- load_json('train.json')
test  <- load_json('test.json')
train[1:5]
train[, ingredients := lapply(ingredients, tolower)]
test[, ingredients := lapply(ingredients, tolower)]
extract_text <- function() {
  mapply(function(ingredients, cuisine) {
    x <- paste(sample(ingredients), collapse=sprintf('#%s#', cuisine))
    x <- paste(cuisine, x, cuisine, sep='#')
    trimws(x)
  }, train$ingredients, train$cuisine) 
}
set.seed(0)
text <- do.call(c, replicate(100, extract_text(), simplify=F))
head(text, 3)
special_tokenizer <- function(x, ...) space_tokenizer(x, sep = "#", ...)
it <- itoken(text, tokenizer = special_tokenizer)
vocab <- create_vocabulary(it)
vectorizer <- vocab_vectorizer(vocab)

