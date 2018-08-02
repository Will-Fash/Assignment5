from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from data import DataSet
from show import showChart
import random


def getSVDChart():
    data_set = DataSet()
    data, label, wm = data_set.get_train_data_set()
    indexs = random.sample(range(len(data)), 28000)
    data = data[indexs]
    label = label[indexs]
    count_vect = CountVectorizer()
    X_train_counts = count_vect.fit_transform(data)

    truncatedSVD = TruncatedSVD(n_components=5, n_iter=7, random_state=42)
    truncatedSVD = truncatedSVD.fit(X_train_counts)

    X_r = truncatedSVD.transform(X_train_counts)
    showChart(X_r, label, "PCA metric graph", len(X_r), 5000)


getSVDChart()