import numpy as np
import cv2
import argparse



# Funciones auxiliares para conversión de parametros de bboxs
def convertOriginalCoordToVertex(detections):
    """
    Convierto [x,y,w,h,socore] -> [x1,y1,x2,y2,socore], donde.

    Args:
        detections (numpy.array): Resultado de detecciones de tamaño (num, 5), cuyos parametros de la
            segunda dimension son [x, y, w, h, score] respectivamente, donde (x,y) representan las coordenadas 
            del vertice superior derecho del recuadro delimitador de la detección, y (w,h) su alto y ancho respectivos. 
    
    Returns: 
        det (numpy.array): Detecciones de tamaño (num, 5), cuyos parametros de la segunda dimension 
            son [x, y, w, h, score] respectivamente, donde (x1,y1) indican las coordenadas del vertice superior
            izquierda del recuadro delimitador, y (x2,y2) las del vertice inferior derecho.
    """
    det = detections.copy()

    for i in range(len(det)):
        x, y, w, h = det[i,0], det[i,1], det[i,2], det[i,3]
        det[i,0] = x - w     # x1
        det[i,1] = y         # y1
        det[i,2] = x         # x2
        det[i,3] = y + h     # y2

    return det

def convertVertexToOriginalCoord(detections):
    """
    Convierto parametros de detecciones de la forma [x1,y1,x2,y2,score] -> [x,y,w,h,score].
    """
    det = detections.copy()
    for i in range(len(det)):
        x1, y1, x2, y2 = det[i,0], det[i,1], det[i,2], det[i,3]
        det[i,0] = x2        # x
        det[i,1] = y1        # y
        det[i,2] = x2 - x1   # w
        det[i,3] = y2 - y1   # h

    return det



# Funcion auxiliar para visualización de un ejemplo de ejecucion
def drawDetections(detections, img_path):
    """
    Muestro todas los recuadors delimitadores de las detecciones <detections>, realizadas sobre la imagen <img_path>.

    Args:
        detections (numpy.array): Resultado de detecciones de tamaño (num, 5), cuyos parametros de la
            segunda dimension son [x, y, w, h, score] respectivamente, donde (x,y) representan las coordenadas 
            del vertice superior derecho del recuadro delimitador de la detección, y (w,h) su alto y ancho respectivos. 

        img_path: Ruta de la imagen en donde fueron realizadas el conjunto de detecciones <detections>.
    """

    origin_img = cv2.imread(img_path)
    new_img = origin_img.copy()

    detections = convertOriginalCoordToVertex(detections)

    for i in range(len(detections)):
        x1 = detections[i,0]
        y1 = detections[i,1]
        x2 = detections[i,2]
        y2 = detections[i,3]
        
        color = (0,0,255)
        cv2.rectangle(new_img,(x1,y1),(x2,y2), color, 6)

    cv2.namedWindow('Imagen',cv2.WINDOW_NORMAL)
    cv2.imshow("Imagen", new_img)
    cv2.waitKey(0)



# Funcion para el calculo de indice Intersection over Union (IoU)
def getIou(box1, box2):
    """ 
    Implementacion de algoritmo Intersections Over Union.
    
    Args:
        box1, box2 (numpy.array): Parametros de recuadro delimitador de dimension (4),
            de la forma  [x1, y1, x2, y2] respectivamente.

    Returns:
        float: Valor de superposicion entre ambos recuadros delimitadores.
    """

    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0], box1[1], box1[2] , box1[3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0], box2[1], box2[2] , box2[3]
    
    # Calculo intersecciones
    rect_x1 = np.maximum(box1_x1,box2_x1)
    rect_y1 = np.maximum(box1_y1,box2_y1)
    rect_x2 = np.minimum(box1_x2,box2_x2)
    rect_y2 = np.minimum(box1_y2,box2_y2)

    inter_area = np.maximum(rect_x2 - rect_x1 ,0 ) * np.maximum(rect_y2 - rect_y1 ,0)

    # Calculo areas de cada bbox
    box1_area = (box1_x2 - box1_x1 ) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    # Calculo indice IoU
    iou = inter_area / (box1_area + box2_area - inter_area)

    return iou


def nonMaxSuppression(detections, conf_threshold, iou_threshold):
    """ 
    Implementacion de algoritmo greedy tradicional NMS.
    
    Args:
        detections (numpy.array): Resultado de detecciones de tamaño (num, 5),
            donde los parámetros de la segunda dimensionn son [x, y, w, h, score] respectivamente.
        
        conf_threshold (float): Umbral para descartar detecciones con bajos valores de confidencia. 

        iou_threshold (float): Umbral utilizado para descartar superpociciones de detecciones.
            maximum > thresh.

    Returns:
        numpy.array: Detecciones retenidas.
    """

    final_detections = []
    detections = convertOriginalCoordToVertex(detections)

    # Ordeno detecciones segun indice de confidencia decreciente (score).
    idx = np.argsort(detections[:,4])[::-1]
    sorted_detections = detections[idx]

    # Descarto detecciones que posean un valor de confidencia (score), 
    # menor a conf_threshold. 
    mask = (sorted_detections[:,4] >= conf_threshold)
    sorted_detections = sorted_detections[mask]


    while (sorted_detections.shape[0] > 0):

        # 1. Tomo deteccion con mayor indice de confidencia
        best_det = sorted_detections[0]
        final_detections.append(best_det)

        sorted_detections = sorted_detections[1:]
    
        # 2. calculo IoU entre deteccion seleccionada en 1. y las restantes,
        # eliminando aquellas que tengan un IoU >= iou_threshold           
        idx_to_reteined = []
        for j , current_det in enumerate(sorted_detections):
            iou = getIou(best_det[:4], current_det[:4])
            if (iou < iou_threshold):
                idx_to_reteined.append(j)

        sorted_detections = sorted_detections[idx_to_reteined]

        
    final_detections = convertVertexToOriginalCoord(np.array(final_detections))
    return final_detections



# Funcion para recuperar argumentos de linea de comando
def getArguments():
    ap = argparse.ArgumentParser(description='Implementacion de algoritmo greedy tradicional NMS.')
    ap.add_argument("-conf", "--conf_threshold", default=0.5, type=float, help="Valor umbral para descartar detecciones con baja confidencia. (Default = 0.5)")
    ap.add_argument("-iou", "--iou_threshold", default=0.45, type=float, help="Valor umbral utilizado para filtrar valores elevados de superposiciones de bboxs. (Default: 0.45)")
    return vars(ap.parse_args())  


# Test
def test(conf_threshold, iou_threshold):

    image_path = "test.jpg"
    
    detections = np.array([
                    [1110, 519, 467, 515, 0.9],
                    [1833, 666, 599, 280, 0.88],
                    [1012, 493, 395, 444, 0.5],
                    [1114, 453, 408, 519, 0.6],
                    [1079, 595, 418, 413, 0.8],
                    [1838, 639, 600, 200, 0.67],
                    [1798, 719, 591, 227, 0.5],
                    [1869, 670, 560, 236, 0.67]
                ], dtype=np.float32)

    # Muestro deteccionesde de prueba
    print("\nLista de detecciones. Pulse ENTER sobre la imagen para correr NMS...")
    print(np.vstack(detections))
    drawDetections(detections, image_path)

    # Ejecuto algoritmo NMS
    final_detections = nonMaxSuppression(detections, conf_threshold, iou_threshold)
    print("\nResultado de NMS:")
    print(final_detections)
    drawDetections(final_detections, image_path)



if __name__ == '__main__':

    args = getArguments()
    test(args["conf_threshold"], args["iou_threshold"])