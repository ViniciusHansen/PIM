import cv2
import numpy as np

# Carregar a imagem
imagem_rgb = cv2.imread('/home/hnz/PIM/P1/Lighthouse_rgb.png') 
if imagem_rgb is None:
    print("Erro ao carregar a imagem!")
    exit()

# Obter as dimensões da imagem
altura, largura, _ = imagem_rgb.shape

# Criar uma matriz vazia para a imagem em tons de cinza
imagem_gray = np.zeros((altura, largura), dtype=np.uint8)

# Converter a imagem RGB para tons de cinza usando a fórmula
for i in range(altura):
    for j in range(largura):
        r = imagem_rgb[i, j, 2]
        g = imagem_rgb[i, j, 1]
        b = imagem_rgb[i, j, 0]
        gray = int(0.299 * r + 0.587 * g + 0.114 * b)
        imagem_gray[i, j] = gray



def isodata_thresholding(img_gray):
    # Passo 1: Escolha um valor inicial para o limiar T
    limiar = np.mean(img_gray)
    limiar_prev = 0
    
    while abs(limiar - limiar_prev) > 0.5:  # Continue até que a mudança no limiar seja menor que 0.5
        limiar_prev = limiar
        
        # Passo 2: Divida a imagem em dois grupos
        below_threshold = img_gray[img_gray < limiar]
        above_threshold = img_gray[img_gray >= limiar]
        
        # Passo 3: Calcule a média dos pixels para cada grupo
        mean_below = np.mean(below_threshold)
        mean_above = np.mean(above_threshold)
        
        # Passo 4: Calcule um novo limiar T
        T = (mean_below + mean_above) / 2

    return T

limiar = isodata_thresholding(imagem_gray)

# Binarizar a imagem usando o limiar obtido
_, imagem_binarizada = cv2.threshold(imagem_gray, limiar, 255, cv2.THRESH_BINARY)

cv2.imwrite('Lighthouse_gray_manual.png', imagem_gray)
cv2.imshow('Imagem em Tons de Cinza Manual', imagem_gray)
cv2.imshow('Imagem Binarizada', imagem_binarizada)
cv2.waitKey(0)
cv2.destroyAllWindows()
