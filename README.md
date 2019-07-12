**Problem Statement**
The training dataset consists of different recipes distinguished by Recipe_id. Each recipe has a list of ingredients and the cuisine which it belongs to. The objective is to develop a model from the given dataset which classifies a recipe’s cuisine given its ingredients.

**Data  Understanding**
1.)The dataset is in JSON format. The dependent variable is cuisine and independent variable is ingredients.     
2.)Train data consists of 39774 dishes as dish-ids, ingredients and the cuisine they belong to.
3.)Test data consists of 9944 dishes as dish-ids and ingredients.
4.)There are 20 different unique cuisines present in the data
5.)There are total 428275 number of ingredients among which 6703 are unique
6.)The dataset  does not have many attributes.In our dataset the main attribute  is a list of set of ingredients.Each set represents a dish.The data is basically categorical data and data types is list of strings
