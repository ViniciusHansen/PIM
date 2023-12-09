import cv2
import numpy as np

# Carregar a imagem
imagem = cv2.imread('/home/hnz/PIM/Imagens-teste/Lighthouse_bayerBG8.png', cv2.IMREAD_GRAYSCALE)  # A imagem Bayer é monocromática
if imagem is None:
    print("Erro ao carregar a imagem!")
    exit()

# Acrescentar a moldura de zeros
imagem_padded = cv2.copyMakeBorder(imagem, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)

# Criar matrizes vazias para os canais R, G e B
R = np.zeros_like(imagem_padded)
G = np.zeros_like(imagem_padded)
B = np.zeros_like(imagem_padded)

# Preencher os valores conhecidos dos canais R, G e B
for i in range(1, imagem_padded.shape[0] - 1):
    for j in range(1, imagem_padded.shape[1] - 1):
        if i % 2 == 0:  # Linhas pares
            if j % 2 == 0:  # Colunas pares
                R[i, j] = imagem_padded[i, j]
            else:  # Colunas ímpares
                G[i, j] = imagem_padded[i, j]
        else:  # Linhas ímpares
            if j % 2 == 0:  # Colunas pares
                G[i, j] = imagem_padded[i, j]
            else:  # Colunas ímpares
                B[i, j] = imagem_padded[i, j]

# Converta as matrizes para int32 para evitar overflow
R = R.astype(np.int32)
G = G.astype(np.int32)
B = B.astype(np.int32)

# Realizar a interpolação para os valores desconhecidos
# Para simplificar, usaremos a média dos vizinhos para interpolar
for i in range(1, imagem_padded.shape[0] - 1):
    for j in range(1, imagem_padded.shape[1] - 1):
        if R[i, j] == 0:
            R[i, j] = (R[i-1, j] + R[i+1, j] + R[i, j-1] + R[i, j+1]) // 4
        if G[i, j] == 0:
            G[i, j] = (G[i-1, j] + G[i+1, j] + G[i, j-1] + G[i, j+1]) // 4
        if B[i, j] == 0:
            B[i, j] = (B[i-1, j] + B[i+1, j] + B[i, j-1] + B[i, j+1]) // 4

# Remover a moldura
R = R[1:-1, 1:-1]
G = G[1:-1, 1:-1]
B = B[1:-1, 1:-1]

# Converte de volta para utin8
R = np.clip(R, 0, 255).astype(np.uint8)
G = np.clip(G, 0, 255).astype(np.uint8)
B = np.clip(B, 0, 255).astype(np.uint8)

# Combinar os canais R, G e B para formar a imagem RGB
imagem_rgb = cv2.merge((B, G, R))

# Exibir e salvar a imagem RGB
cv2.imshow('Imagem RGB', imagem_rgb)
#cv2.imwrite('Lighthouse_rgb.png', imagem_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
