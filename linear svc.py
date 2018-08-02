from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from data import DataSet
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
from conf_lib import plot_confusion_matrix
from sklearn.decomposition import TruncatedSVD
import random

data_set = DataSet()
data, label, class_names = data_set.get_train_data_set()

indexs = random.sample(range(len(data)), 45000)
data = data[indexs]
label = label[indexs]
X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.25, random_state=42)

est = [('count_vect',CountVectorizer()),('tr',TruncatedSVD(n_components=60, n_iter=300, random_state=42)),('clf_SVC', LinearSVC(C=0.75, random_state=0, max_iter=500))]

pipeline_SVC = Pipeline(est)


pipeline_SVC = pipeline_SVC.fit(X_train, y_train)
y_pred = pipeline_SVC.predict(X_test)
print("F1 score - SVC:", f1_score(y_test, y_pred, average='micro'))
print("Accuracy Score - SVC:", accuracy_score(y_test, y_pred))
cnf_matrix = confusion_matrix(y_test, y_pred)
plt.figure()
plt = plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                            title='Normalized confusion matrix SVC')
plt.show()