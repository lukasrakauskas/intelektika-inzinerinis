from sklearn_som.som import SOM
from sklearn import datasets
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import pandas as pd
import numpy as np

df = pd.read_csv('heart.csv')
age = df['age'].to_numpy()
thalachh = df['thalachh'].to_numpy()
data = np.vstack((age, thalachh)).T

#iris = datasets.load_iris()
#iris_data = iris.data[:, :2]
#iris_label = iris.target
iris_som = SOM(m=5, n=1, dim=2)
iris_som.fit(data)
predictions = iris_som.predict(data)
# Plot the results
#fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(5, 7))
x = data[:, 0]
y = data[:, 1]
colors = ['red', 'green', 'blue', 'purple', 'yellow']

# ax[0].scatter(x, y, cmap=ListedColormap(colors))
# ax[0].title.set_text('Actual Classes')
plt.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
plt.title('SOM Predictions')
plt.show()
