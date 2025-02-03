import numpy as np

class lineal_regression():
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N = len(X)
        self.mean_x = np.mean(self.X)
        self.mean_y = np.mean(self.Y)
        self.sum_x = self.X.sum()
        self.sum_y = self.Y.sum()

        self.Sxy = sum((x_i - self.mean_x) * (y_i - self.mean_y) for x_i, y_i in zip(self.X, self.Y))
        self.Sxx = sum((x_i - self.mean_x)**2 for x_i in self.X)
        self.Syy = sum((y_i - self.mean_y)**2 for y_i in self.Y)

        self.r = self.Sxy/np.sqrt(self.Sxx*self.Syy)
        self.r2 = self.r**2

        self.b1 = self.Sxy/self.Sxx
        self.b0 = self.mean_y - self.b1*self.mean_x

        self.sce = self.Syy - ((self.Sxy)**2)/self.Sxx
        self.stc = self.Syy
        self.scr = self.stc - self.sce

        self.var = self.sce/(self.N - 2)

    def predict(self):
        aux = []
        for value in self.X:
            aux.append(self.b0 + self.b1*value)
        return aux