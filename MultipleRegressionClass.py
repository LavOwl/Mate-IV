import numpy as np

class multiple_regression_class():

    def __init__(self, Xs, Y):
        self.Xs = Xs
        self.Y = Y
        self.N = len(self.Y)

        self.sum_Y = self.Y.sum()
        self.mean_y = np.mean(self.Y)
        self.stc = sum((y_i - self.mean_y)**2 for y_i in self.Y)

        def calculate_sums():
            aux = []
            for x in self.Xs:
                aux.append(x.sum())
            return aux
        
        self.sum_Xs = calculate_sums()
        
        def create_y_vector():
            aux = []
            v = []
            v.append(self.sum_Y)
            aux.append(v)
            for x in self.Xs:
                v = []
                v.append(sum(x_i*y_i for x_i, y_i in zip(x, self.Y)))
                aux.append(v)
            return aux
        
        def create_x_matrix():
            matrix = []
            aux = []
            aux.append(self.N)
            for x in self.sum_Xs:
                aux.append(x)
            matrix.append(aux)
            for i in range(len(self.Xs)):
                aux = []
                aux.append(self.sum_Xs[i])
                for j in range(len(self.Xs)):
                    aux.append(special_sum(self.Xs[i], self.Xs[j]))
                matrix.append(aux)
            return matrix
        
        def special_sum(equis, otroEquis):
            return (sum(x_i*ex_i for x_i, ex_i in zip(equis, otroEquis)))
        
        def multiply_matrices(A, B):
            matrix = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
            for i in range(len(A)):
                for j in range(len(B[0])):
                    for k in range(len(B)):
                        matrix[i][j] += A[i][k] * B[k][j]
            return matrix
        
        def matrix_inverse(matrix):
            n = len(matrix)

            identity_matrix = [[float(i == j) for i in range(n)] for j in range(n)]

            for i in range(n):
                diag_element = matrix[i][i]
                
                for j in range(n):
                    matrix[i][j] /= diag_element
                    identity_matrix[i][j] /= diag_element

                for k in range(n):
                    if k != i:
                        factor = matrix[k][i] #Siempre queda en 0
                        for j in range(n):
                            matrix[k][j] -= factor * matrix[i][j] #Cuando i == j matrix[k][j] == matrix[k][i], que es factor, y matrix[i][j] == matrix[i][i] == 1, por tanto -factor*1 anula la matrix
                            #Vale recalcar que el algoritmo nunca modifica un 0 o un 1 que ya estuviera correctamente configuardo ya que a cada fila se le resta una fila donde m[i][j] es 1, y por tanto lo que se resta a su izquierda está multiplicado por 0, o no se multiplica a su izquierda.
                            identity_matrix[k][j] -= factor * identity_matrix[i][j]

            return identity_matrix

        self.coefficients = multiply_matrices(matrix_inverse(create_x_matrix()), create_y_vector())

        def simple_predict(exs):
            aux = float(self.coefficients[0][0])
            for i in range(len(exs)):
                aux += exs[i]*float(self.coefficients[i+1][0])
            return aux

        def calculate_sce():
            aux = 0
            for i in range(self.N):
                exs = []
                for j in range(len(self.Xs)):
                    exs.append(self.Xs[j][i])
                aux += (self.Y[i] - simple_predict(exs))**2
            return aux

        self.sce = calculate_sce()
        self.r2 = 1 - self.sce/self.stc
        self.r2a = 1 - (1 - self.r2)*((self.N - len(self.Xs))/(self.N - len(self.Xs) - 1))