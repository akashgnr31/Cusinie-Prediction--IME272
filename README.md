**Problem Statement**
The training dataset consists of different recipes distinguished by Recipe_id. Each recipe has a list of ingredients and the cuisine which it belongs to. The objective is to develop a model from the given dataset which classifies a recipe’s cuisine given its ingredients.

**Data  Understanding**
1.)The dataset is in JSON format. The dependent variable is cuisine and independent variable is ingredients.   

2.)Train data consists of 39774 dishes as dish-ids, ingredients and the cuisine they belong to.

3.)Test data consists of 9944 dishes as dish-ids and ingredients.

4.)There are 20 different unique cuisines present in the data

5.)There are total 428275 number of ingredients among which 6703 are unique

6.)The dataset  does not have many attributes.In our dataset the main attribute  is a list of set of ingredients.Each set represents a dish.The data is basically categorical data and data types is list of strings


**Data Cleaning:**
The given dataset contain some quirks and needs to be cleaned.We took the following steps in Data Cleaning-

a).Checked the presence of missing values which came out to be NIL.

b).The incoming data can be loosely structured, multilingual, textual or might have poor spelling for example:

Some ingredients may contain special symbols that are not relevant like @,# etc and are needed to be removed.

Since text cases doesn’t change the meaning of the word so its better to convert them into single lower case.

Punctuation elements like “ “ , - etc are not useful and shall be removed .For ex, ‘Chilli Flakes’ and chilli-flakes are same.

Stopwords are those words which add no value .They don’t describe any sentiment. Examples are ‘i’,’me’,’myself’,’they’,’them’ and many more. Hence, these words should be removed if present.

Stemming means bringing a word back to its roots. It is generally used for words which are similar but only differ by tenses.To treat word and word derivatives as same we stemmed our dataset.

c).Checked and removed duplicate data which resulted in numerosity reduction.

d).Set the ingredient data to unique for each row.

Ideally, we have done a Data cleaning such that it returned a more relevant version of these ingredients.

**Data Reduction and Transformation :**
a.)Combined the ingredients of a row  to form a text document which gave a data set of about 39,774 documents with a class label attached to it.

b.)Removed the Duplicate rows present in Data which resulted in a little  of numerosity Reduction.

Data Reduced: 39774 to 39256

c.)Document Term Matrix:

From the text data (set of ingredients), we have created document term matrix (DTM) which assigns 1 to a ingredients present in a dish and 0 to all those not present in the dish.

‘Since DTM was a Sparse matrix so we tried to reduce it’s sparseness by removing the ingredients with frequency less than 3 in whole dataset.

d.)TF-IDF Matrix:

This is a technique to quantify a word in documents, The  weights to each word are computed which signifies the importance of the word in the document and corpus with respect to other documents. 

It gives more weight to those ingredients which occurs more often in one class of documents given it is less present in other classes of documents.

From the DTM matrix we get the tf(Term frequency) and IDF(Inverse Document frequency) of each word in a document and multiplied them to obtain the TF-IDF value of a word. 

We assigned these tf-idf value to each term in document and formed a matrix.
Reduced sparse number of columns from 6703 ingredients to 3010.

The TF-IDF matrix will be the input to most of our Models except in Naive Bayes where DTM is the training data. 

**Data Mining Algorithms Applied**


**1.)Artificial Neural Networks**


**2.)Support Vector Machine**


**3.)Logistic Regression**


**4.)Random Forest**


**5.)Naive Bayes**


**6.)Ensemble Classifier**
