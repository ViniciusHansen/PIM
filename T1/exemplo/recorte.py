import cv2
import numpy as np
import matplotlib.pyplot as plt

def recorta(x, y, offset):
    img = cv2.imread('lenaShort.jpg')
    print(img.shape)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Original')
    plt.show()
    
    coords = []
    for linha in range((x-offset),(x+offset)):
        for coluna in range ((y-offset),(y+offset)):
            coordenada =  (linha, coluna)
            coords.append(coordenada)
    print(coords)


    media, dp = cv2.meanStdDev(img)
    media = media[0][0]
    dp = dp[0][0]
    print(f"Média: {media}")
    print(f"Desvio Padrão: {dp}")

    cropped_image = img[(x-offset):(x+offset), (y-offset):(y+offset)]
    plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    plt.title('Cropped')
    plt.show()

recorta(100, 100, 80)
