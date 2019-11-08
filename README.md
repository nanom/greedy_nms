# greedy_nms
## Implementación tradicional de algoritmo greedy Non Max Suppression
El archivo nms.py, contiene tanto la imlementación de la función encargada de realizar NMS 'nonMaxSuppression()', como de otras funciones auxiliares que fueron utilizadas para poder visualizar el funcionameinto de tal algoritmo sobre una imagen 'test.jpg', y un conjunto de detecciones realizadas sobre la misma.

### Modo de uso
El script, ya cuenta con la implemetada de la funcion 'test()', la cual despliga una ventana con la imagen y todas las detecciones realizadas sobre la misma, tanto antes, como despues de ejecutarse el algoritmo de NMS, automaticamente. A continuación se describe en detalle su forma de uso.
```
usage: python nms.py [-h] [-conf CONF_THRESHOLD] [-iou IOU_THRESHOLD]

optional arguments:
  -h, --help            show this help message and exit
  -conf CONF_THRESHOLD, --conf_threshold CONF_THRESHOLD
                        Valor umbral para descartar detecciones con baja
                        confidencia. (Default = 0.5)
  -iou IOU_THRESHOLD, --iou_threshold IOU_THRESHOLD
                        Valor umbral utilizado para filtrar valores elevados
                        de superposiciones de bboxs. (Default: 0.45)
```