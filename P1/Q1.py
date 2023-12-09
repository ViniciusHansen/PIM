import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('../Imagens-teste/blue.png')

# Separar os canais RGB
B, G, R = cv2.split(imagem)


# Calcular a média de cada canal
mean_B = np.mean(B)
mean_G = np.mean(G)
mean_R = np.mean(R)

# Verificar qual canal tem a maior média
if mean_B > mean_G and mean_B > mean_R:
    print("A imagem é dominada pela cor azul.")
elif mean_G > mean_B and mean_G > mean_R:
    print("A imagem é dominada pela cor verde.")
elif mean_R > mean_B and mean_R > mean_G:
    print("A imagem é dominada pela cor vermelha.")
elif np.all(imagem == 0):
    print("A imagem é preta.")
elif np.all(imagem == 255):
    print("A imagem é branca.")
else:
    print("A imagem tem uma outra combinação de cores.")