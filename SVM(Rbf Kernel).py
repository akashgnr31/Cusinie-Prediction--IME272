from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow import keras
import pandas as pd
import json
import numpy as np

def read_dataset(path):
    return json.load(open(path)) 

train = read_dataset('../IME672/train.json')
test = read_dataset('../IME672/test.json')

def generate_text(data):
    text_data = [" ".join(doc['ingredients']).lower() for doc in data]
    return text_data 
    
train_text = generate_text(train)
test_text = generate_text(test)
target = [doc['cuisine'] for doc in train]

tfidf = TfidfVectorizer(binary=True)
def tfidf_features(txt, flag):
    if flag == "train":
        x = tfidf.fit_transform(txt)
    else:
        x = tfidf.transform(txt)
    x = x.astype('float16')
    return x 
X = tfidf_features(train_text, flag="train")
X_test = tfidf_features(test_text, flag="test")

lb = LabelEncoder()
y = lb.fit_transform(target)
print ("Train the model ... ")
classifier = SVC(C=100, # penalty parameter
	 			 kernel='rbf', # kernel type, rbf working fine here
	 			 degree=3, # default value
	 			 gamma=1, # kernel coefficient
	 			 coef0=1, # change to 1 from default value of 0.0
	 			 shrinking=True, # using shrinking heuristics
	 			 tol=0.001, # stopping criterion tolerance 
	      		 probability=False, # no need to enable probability estimates
	      		 cache_size=200, # 200 MB cache size
	      		 class_weight=None, # all classes are treated equally 
	      		 verbose=False, # print the logs 
	      		 max_iter=-1, # no limit, let it run
          		 decision_function_shape=None, # will use one vs rest explicitly 
          		 random_state=None)
model_SVM = OneVsRestClassifier(classifier, n_jobs=1)

X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2 ,random_state = 0)

st = time.time()
model_SVM.fit(X_train, y_train)

y_predict_SVM = model_SVM.predict(X_test)
y_predict_SVM=lb.inverse_transform(y_predict_SVM)
y_predict_SVM = lb.fit_transform(y_predict_SVM)

print('Accuracy: %0.4f [SVM RBF] [Time: %ss]' %(accuracy_score(y_test, y_predict_SVM), (time.time()-st)))

summary = np.zeros((20, 20), dtype=np.int32)
for y_test_i, y_predict_i in zip(y_test, y_predict_SVM):
    summary[y_test_i, y_predict_i] += 1

summary_df = pd.DataFrame(summary, 
                          columns=cuisines, 
                          index=cuisines)

summary_df

summary_norm = ( summary / summary.sum(axis=1) )
sns.heatmap( summary_norm, 
            vmin=0, vmax=1, center=0.5, 
            xticklabels=cuisines,
            yticklabels=cuisines);
            
print(classification_report(y_test,y_predict_SVM))
