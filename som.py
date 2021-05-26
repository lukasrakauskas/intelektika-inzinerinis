import numpy as np  # pip install numpy
import pandas as pd  # pip install pandas
import matplotlib.pyplot as plt  # pip install matplotlib
from matplotlib.colors import ListedColormap  # pip install numpy


class SOM:
    def __init__(self, m=3, n=3, dim=3, lr=1, sigma=1, max_iter=3000):
        # Pasikuriam reikiamus kintamuosius
        self.m = m
        self.n = n
        self.dim = dim
        self.shape = (m, n)
        self.initial_lr = lr
        self.lr = lr
        self.sigma = sigma
        self.max_iter = max_iter

        # Pasikuriam svorius
        self.weights = np.random.normal(size=(m * n, dim))
        self._locations = self._get_locations(m, n)

    def _get_locations(self, m, n):
        return np.argwhere(np.ones(shape=(m, n))).astype(np.int64)

    def _find_bmu(self, x):
        # Kiekvienam svoriui bus po 1 eile
        x_stack = np.stack([x]*(self.m*self.n), axis=0)
        # Paskaiciuojam atstuma tarp x ir kiekvieno svorio
        distance = np.linalg.norm(x_stack - self.weights, axis=1)
        # Randam labiausiai tinkamo vieneto indeksa
        return np.argmin(distance)

    def step(self, x):
        # Kiekvienam svoriui bus po 1 eile
        x_stack = np.stack([x]*(self.m*self.n), axis=0)

        # Randam labiausiai tinkamo vieneto indeksa
        bmu_index = self._find_bmu(x)

        # Randam labiausiai tinkamo vieneto vieta
        bmu_location = self._locations[bmu_index, :]

        # Randam kvadratini atstuma iki kiekvieno labiausiai tinkamo vieneto
        stacked_bmu = np.stack([bmu_location]*(self.m*self.n), axis=0)
        bmu_distance = np.sum(np.power(self._locations.astype(
            np.float64) - stacked_bmu.astype(np.float64), 2), axis=1)

        # Atnaujinam kaimynus
        neighborhood = np.exp((bmu_distance / (self.sigma ** 2)) * -1)
        local_step = self.lr * neighborhood

        # Padarome zingsni reikiamos formos
        local_multiplier = np.stack([local_step]*(self.dim), axis=1)

        # Padauginam is zingsnio ir svoriu skirtumo
        delta = local_multiplier * (x_stack - self.weights)

        # Atnaujinam svorius
        self.weights += delta

    def fit(self, X, epochs=1, shuffle=True):
        # Paskaiciuojame iteraciju kieki
        global_iter_counter = 0
        n_samples = X.shape[0]
        total_iterations = np.minimum(epochs * n_samples, self.max_iter)

        for epoch in range(epochs):
            # Stabdom jei baigesi iteraciju kiekis
            if global_iter_counter > self.max_iter:
                break

            if shuffle:
                indices = np.random.permutation(n_samples)
            else:
                indices = np.arange(n_samples)

            # Apsimokymas
            for idx in indices:
                # Stabdom jei baigesi iteraciju kiekis
                if global_iter_counter > self.max_iter:
                    break
                input = X[idx]
                # Darome 1 treniravimo zingsni
                self.step(input)
                # Atnaujinam mokymosi greiti
                global_iter_counter += 1
                self.lr = (1 - (global_iter_counter /
                                total_iterations)) * self.initial_lr

    def predict(self, X):
        labels = np.array([self._find_bmu(x) for x in X])
        return labels


som = SOM(m=5, n=1, dim=2)
df = pd.read_csv('heart.csv')
age = df['age'].to_numpy()
thalachh = df['thalachh'].to_numpy()
data = np.vstack((age, thalachh)).T
som.fit(data)
predictions = som.predict(data)
x = data[:, 0]
y = data[:, 1]
colors = ['red', 'green', 'blue', 'purple', 'yellow']

plt.scatter(x, y, c=predictions, cmap=ListedColormap(colors))
plt.title('SOM Predictions')
plt.show()
