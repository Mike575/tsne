from sklearn import datasets
from openTSNE import TSNE

import utils

import numpy as np
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

iris = datasets.load_iris()
x, y = iris["data"], iris["target"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.33, random_state=42)
print("%d training samples" % x_train.shape[0])
print("%d test samples" % x_test.shape[0])

tsne = TSNE(
    perplexity=30,
    metric="euclidean",
    n_jobs=8,
    random_state=42,
    verbose=True,
)
embedding_train = tsne.fit(x_train)
utils.plot(embedding_train, y_train, colors=utils.MOUSE_10X_COLORS)
embedding_test = embedding_train.transform(x_test)
utils.plot(embedding_test, y_test, colors=utils.MOUSE_10X_COLORS)

fig, ax = plt.subplots(figsize=(8, 8))
utils.plot(embedding_train, y_train, colors=utils.MOUSE_10X_COLORS, alpha=0.25, ax=ax)
utils.plot(embedding_test, y_test, colors=utils.MOUSE_10X_COLORS, alpha=0.75, ax=ax)
plt.show()