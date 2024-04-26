import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectKBest, mutual_info_classif

df = pd.read_csv('harth_S028.csv')
df.head()

df = df.drop('timestamp', axis =1)

target = df['label']
df = df.drop('label', axis  =1)

df.describe()

scaler = StandardScaler()
df_standardized = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
df_standardized.describe()

Xtrain, Xtest, Ytrain, Ytest = train_test_split(df_standardized, target, test_size=0.2, random_state=43)

knn = KNeighborsClassifier()
knn.get_params()

knn.fit(Xtrain, Ytrain)
predict = knn.predict(Xtest)

cm = confusion_matrix(Ytest, predict)
report = classification_report(Ytest, predict, digits =4)
sb.heatmap(cm, cmap="Blues", annot=True, fmt='d',
           xticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'],
           yticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'])
plt.xlabel("Prediced")
plt.ylabel("Actual")
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Confusion Matrix for best weights assigned for best 4 correlated features", fontweight='bold')
print(report)

tempDf = pd.concat([df,target], axis = 1)
corMat = tempDf.corr()
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Correlation Matrix", fontweight='bold')
sb.heatmap(corMat, cmap = 'Blues', annot = True)
plt.show()

corrValues = corMat['label']
plt.figure(figsize=(6, 5))
sb.barplot(x=corrValues.index, y=corrValues.values, color='b')
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Correlation with Target Variable", fontweight='bold')
plt.xlabel('Correlation')
plt.xticks(rotation='vertical')
plt.ylabel('Features')

np.random.seed(10)
weights = np.random.random(6)
wKnn = KNeighborsClassifier(metric = 'wminkowski', p = 2, metric_params = {'w':weights})

wKnn.fit(Xtrain, Ytrain)
Wpredict = wKnn.predict(Xtest)

cm = confusion_matrix(Ytest, Wpredict)
report = classification_report(Ytest, Wpredict, digits =4)
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Confusion Matrix for default parameters in kNN Classifier", fontweight='bold')
sb.heatmap(cm, cmap="Blues", annot=True, fmt='d',
           xticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'],
           yticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'])
plt.xlabel("Prediced")
plt.ylabel("Actual")
print(report)

temp= corMat.iloc[-1,:-1]
high_corr = temp.nlargest(4).index
high_index = temp.index.get_indexer(high_corr)

new_weights = np.copy(weights)
new_weights[high_index] = [4, 3, 2, 1]

tKnn = KNeighborsClassifier(metric = 'wminkowski', p = 2, metric_params = {'w':new_weights})
tKnn.fit(Xtrain, Ytrain)
tpredict = tKnn.predict(Xtest)

cm = confusion_matrix(Ytest, Wpredict)
report = classification_report(Ytest, tpredict, digits =4)
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Confusion Matrix for random generated weights", fontweight='bold')
sb.heatmap(cm, cmap="Blues", annot=True, fmt='d',
           xticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'],
           yticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'])
plt.xlabel("Prediced")
plt.ylabel("Actual")
print(report)

# Search space for k (from 1 to 30 with a step size of 2)
k_values = np.arange(1, 31, 2)

# Initialize a dictionary to store the mean accuracy for each k
k_accuracy = {}

# Perform k-fold cross-validation for each value of k
k_fold = 10  # Number of folds for cross-validation
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, Xtrain, Ytrain, cv=k_fold, scoring='accuracy')
    k_accuracy[k] = scores.mean()

# Find the optimum value of k with the highest accuracy
best_k = max(k_accuracy, key=k_accuracy.get)
best_accuracy = k_accuracy[best_k]

print(f"\nBest k: {best_k}, Best Mean Accuracy: {best_accuracy}")

plt.plot(k_accuracy.keys(), k_accuracy.values(), marker='o')
plt.xlabel('k Value')
plt.ylabel('Mean Accuracy')
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Accuracy vs. k Value for k-Nearest Neighbors", fontweight='bold')
plt.xticks(k_values)
plt.grid(True)
plt.show()

knn_classifier = KNeighborsClassifier(n_neighbors=best_k)
knn_classifier.fit(Xtrain, Ytrain)
opredict = knn_classifier.predict(Xtest)

cm = confusion_matrix(Ytest, opredict)
report = classification_report(Ytest, opredict, digits =4)
print(report)
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Confusion Matrix for best k-fold value in kNN", fontweight='bold')
sb.heatmap(cm, cmap="Blues", annot=True, fmt='d',
           xticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'],
           yticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'])
plt.xlabel("Prediced")
plt.ylabel("Actual")
plt.show()

best_knn = KNeighborsClassifier()

# Perform feature selection using Mutual Information (SelectKBest)
num_features = 4  # Number of features to select
selector = SelectKBest(score_func=mutual_info_classif, k=num_features)
Xtrain_selected = selector.fit_transform(Xtrain, Ytrain)
Xtest_selected = selector.transform(Xtest)

# Fit kNN classifier on selected features
best_knn.fit(Xtrain_selected, Ytrain)
predict = best_knn.predict(Xtest_selected)

# Generate confusion matrix and classification report for kNN
cm = confusion_matrix(Ytest, predict)
report = classification_report(Ytest, predict, digits=4)
print(report)

# Visualize confusion matrix
sb.heatmap(cm, cmap="Blues", annot=True, fmt='d',
           xticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'],
           yticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Confusion Matrix for kNN Classifier with SelectKBest (Mutual Information)", fontweight='bold')
plt.show()

# Get feature names and their corresponding Mutual Information scores
feature_names = df.columns
mi_scores = selector.scores_

# Plot Mutual Information scores for each feature
plt.figure(figsize=(10, 6))
plt.bar(feature_names, mi_scores, color='blue')
plt.xlabel("Features")
plt.ylabel("Mutual Information Score")
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Mutual Information Scores for Features", fontweight='bold')
plt.xticks(rotation='vertical')
plt.show()

knn_cosine = KNeighborsClassifier(n_neighbors=best_k, metric='cosine')
knn_cosine.fit(Xtrain, Ytrain)
cpredict = knn_cosine.predict(Xtest)

cm = confusion_matrix(Ytest, cpredict)
report = classification_report(Ytest, cpredict, digits =4)
print(report)
plt.title("[THA076BEI040, THA076BEI042]", fontsize=10)
plt.suptitle("Confusion Matrix for best k-fold value in kNN using Cosine Distance", fontweight='bold')
sb.heatmap(cm, cmap="Blues", annot=True, fmt='d',
           xticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'],
           yticklabels=['walking', 'running', 'shuffling', 'stairs (ascending)', 'stairs (descending)', 'standing', 'sitting', 'lying'])
plt.xlabel("Prediced")
plt.ylabel("Actual")
plt.show()