from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.metrics import confusion_matrix


X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

#print(X)
#print(y_true)

#----------------------------------------------------------------------------------------
# TODO determine the best k for k-means

model = KMeans()

visualizer = KElbowVisualizer(model, k=(1,20))
visualizer.fit(X)
visualizer.show()

print("\nBest K: ", visualizer.elbow_value_, "\n")
print("Elbow distortion score: ", visualizer.elbow_score_, "\n")


#----------------------------------------------------------------------------------------
# TODO calculate accuracy for best K

k = visualizer.elbow_value_

kmeans = KMeans(n_clusters=k)
kmeans.fit(X)
y_pred = kmeans.predict(X)

plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis');
plt.title('Original')
plt.show()

plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);
plt.title('Model')
plt.show()

#print("y_true : ", y_true, "\n")
#print("y_pred: ", y_pred, "\n")
print("Accuracy ( k =",k,"): ", accuracy_score(y_true, y_pred), "\n")


#----------------------------------------------------------------------------------------
# TODO draw a confusion matrix
matrix = confusion_matrix(y_true, y_pred)
sns.heatmap(matrix.T, square=True, annot=True, fmt='d', cbar=False,
            xticklabels=[0,1,2,3],
            yticklabels=[0,1,2,3])
plt.xlabel('true label')
plt.ylabel('predicted label');
plt.show()





#----------------------------------------------------------------------------------------
