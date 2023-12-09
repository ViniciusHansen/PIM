import numpy as np


points_3D = [
    [0.0, 1000.0, 200.0, 1],
    [600.0, 1000.0, 400.0, 1],
    [1000.0, 800.0, 400.0, 1],
    [1000.0, 400.0, 600.0, 1],
    [1000.0, 1000.0, 600.0, 1],
    [1000.0, 800.0, 0.0, 1]
]

points_2D = [
    [5, 3, 1],
    [3, 4, 1],
    [2, 4, 1],
    [1, 2, 1],
    [2, 3, 1],
    [2, 5, 1]
]

# Formulate the equations to determine P
A = []

for i in range(6):
    X, Y, Z, W = points_3D[i]
    x, y, w = points_2D[i]
    
    A.append([X, Y, Z, W, 0, 0, 0, 0, -x*X, -x*Y, -x*Z, -x*W])
    A.append([0, 0, 0, 0, X, Y, Z, W, -y*X, -y*Y, -y*Z, -y*W])

A = np.array(A)

# Solve for P using Singular Value Decomposition (SVD)
U, S, Vt = np.linalg.svd(A)
P = Vt[-1].reshape(3, 4)

print(P)

# Projetar os pontos 3D usando P
projected_points = []

for point in points_3D:
    point = np.array(point).reshape(4, 1)
    projected = np.dot(P, point)
    projected /= projected[2]  # Normalizar pela terceira coordenada
    projected_points.append(projected.ravel())

# Calcular o erro entre os pontos projetados e os pontos 2D originais
errors = []

for i in range(len(points_2D)):
    original = np.array(points_2D[i][:2])  # Ignorar a terceira coordenada (homogênea)
    proj = projected_points[i][:2]
    error = np.linalg.norm(original - proj)
    errors.append(error)

# Calcular a acurácia
# Uma abordagem simples é usar o erro médio
mean_error = np.mean(errors)

print("Erro médio:", mean_error)

# A acurácia pode ser definida como 1 - erro médio, mas isso depende da escala dos seus pontos e da aplicação
accuracy = 1 - mean_error

print(f"Acurácia = {accuracy}")
