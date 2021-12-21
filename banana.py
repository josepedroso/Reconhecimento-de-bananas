import cv2
import numpy as np
import sys

def segmentaImagem(nomeImagemEntrada):
    imagemEntrada = cv2.imread(nomeImagemEntrada)
    if imagemEntrada is None:
        print(f"erro na imagem de entrada")
        exit(-1)
    kernel = np.ones((1, 5), 'uint8')
    kernel2 = np.ones((1, 1), 'uint8')
    #filtros pré segmentação
    img_erosion = cv2.erode(imagemEntrada, kernel2, iterations=1)
    dilate_img = cv2.dilate(img_erosion, kernel, iterations=1)
    filtro = cv2.medianBlur(dilate_img, 3)
    #hsv
    hsv = cv2.cvtColor(filtro, cv2.COLOR_BGR2HSV)
    minimo = np.array([16,90,120])#amarelo intervalo minimo em hsv
    maximo = np.array([27,250,255])#amarelo intervalo maximo em hsv

    #Preparando mascara na imagem
    mascara = cv2.inRange(hsv, minimo, maximo)#binaria
    filtro2 = cv2.medianBlur(mascara, 27)#filtro diminuição de ruido
    mascaraFinal = cv2.bitwise_and(imagemEntrada, imagemEntrada, mask=filtro2)#imagem por baixo para aparecer os objetos
    #imagem + mascara
    return np.hstack([imagemEntrada, mascaraFinal])

if __name__ == "__main__":
    if len(sys.argv) == 3:
        for i, arg in enumerate(sys.argv):
            if i == 1:
                img = segmentaImagem(arg)
            if i == 2:
                cv2.imwrite(arg, img)
    else:
        print(f"numero de argumentos errado")