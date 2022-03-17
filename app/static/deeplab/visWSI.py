

import numpy as np
from PIL import Image
import tensorflow as tf
import openslide
from scipy.misc import imsave
from skimage.morphology import remove_small_objects, square, disk, closing, opening
import json
from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize

import matplotlib.pyplot as plt
import cv2
import os


model_path = "datasets/bigdatapath/logdir_train/deeplab_inference_graph.pb"

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
    INPUT_SIZE = 513

    def __init__(self, model_path):
        """Creates and loads pretrained deeplab model."""
        self.graph = tf.Graph()

        graph_def = None
        # Extract frozen graph from tar archive.

        graph_def = tf.GraphDef.FromString(open(model_path, 'rb').read())
        if graph_def is None:
            raise RuntimeError('Cannot find inference graph in tar archive.')
        with self.graph.as_default():
            tf.import_graph_def(graph_def, name='')
        self.sess = tf.Session(graph=self.graph)

    def run(self, image):
        """Runs inference on a single image.
        Args:
            image: A PIL.Image object, raw input image.
        Returns:
            resized_image: RGB image resized from original input image.
            seg_map: Segmentation map of `resized_image`.
        """
        width, height = image.size
        resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
        target_size = (int(resize_ratio * width), int(resize_ratio * height))
        resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
        batch_seg_map = self.sess.run(
            self.OUTPUT_TENSOR_NAME,
            feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
        seg_map = batch_seg_map[0]
        return resized_image, seg_map

def binaryImage2JSON(json_filename, wsi_seg, x=0, y=0):
    im2, contours, hierarchy = cv2.findContours(wsi_seg, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    data = []

    aux = {"applyMatrix": True,
            "segments": [],
            "closed": True,
            "fillColor": [0.61961, 0.61961, 0.61961, 0.5],
            "strokeColor": [0, 0, 0],
            "strokeScaling": False}
    for i, contour in enumerate(contours):
        print (i/len(contours))
        xv = contour[:, 0, 0]
        yv = contour[:, 0, 1]
        data.append({
            'uid': i,
            'name': 'ROI_Carcinoma',
            'imgCoords': [
                    {
                        "x": float(xi + x),
                        "y": float(yi + y)
                    }
                for xi, yi in zip(xv, yv)],
            "path": ["Path", aux],
            "zoom": "0.016597",
            "context": ["IDC"]})    
    if not os.path.isfile(json_filename):        
        with open(json_filename, mode='w') as outfile:
            json.dump(data, outfile)
    else:
        json_data = open(json_filename).read()
        feeds = json.loads(json_data)
        if data:
            feeds = feeds + data

        with open(json_filename, mode='w') as outfile:
            json.dump(feeds, outfile)
                        

#Imagen a segmentar
scan = openslide.OpenSlide("C:/Users/BlancaPT/PycharmProjects/BIGDATA/PI0032WEB/app/static/wsi/397W_HE_40x.svs")

#Dimensiones imagen
width = scan.dimensions[0]
height = scan.dimensions[1]

#Tamaño parches
tile_size = DeepLabModel.INPUT_SIZE

#Stride
stride = np.ceil(tile_size/2).astype(np.long)
stride = tile_size

##Numero de parches
m_tiles = np.fix(np.double(width)/stride)
n_tiles = np.fix(np.double(height)/stride)

#imagen que almacena segmentación
wsi_seg = np.zeros((height, width), bool)

m_vector = np.arange(0, m_tiles).astype(np.long)
n_vector = np.arange(0, n_tiles).astype(np.long)



###############################################################
# Presegmentación (selección de tiles a procesar)
###############################################################

factor_escalado = scan.level_dimensions[0][0]/scan.level_dimensions[scan.level_count-1][0]

x1 = 280 * factor_escalado
x2 = 690 * factor_escalado
y1 = 470 * factor_escalado
y2 = 760 * factor_escalado
m_vector = np.arange(y1 * m_tiles / height, y2 * m_tiles / height).astype(np.long)
n_vector = np.arange(x1 * n_tiles / width, x2 * n_tiles / width).astype(np.long)



#Se lee la WSI con la resolución menor
img_minimum_res = np.array(scan.read_region((0, 0), scan.level_count-1, scan.level_dimensions[scan.level_count-1]))

#Umbralización por color
th = (180, 140, 170)
img_mask = np.logical_and(np.logical_and(img_minimum_res[:, :, 0] < th[0], img_minimum_res[:, :, 1] < th[1]),
                                     img_minimum_res[:, :, 2] < th[2])
#Filtrado ruido
mask_size = 5
clean_img_mask = closing(img_mask, disk(mask_size))  # Closing
clean_img_mask = binary_fill_holes(clean_img_mask, structure=np.ones((mask_size, mask_size)))  # Rellenar huecos
clean_img_mask = remove_small_objects(clean_img_mask, min_size=mask_size**2)  # Eliminar zonas pequeñas

plt.subplot(131), plt.imshow(img_minimum_res)
plt.subplot(132), plt.imshow(img_mask)
plt.subplot(133), plt.imshow(clean_img_mask), plt.show()



model = DeepLabModel(model_path) #Se importa modelo



for i, m in enumerate(m_vector):
    for j, n in enumerate(n_vector):
        n_lowres = np.ceil(n.astype(float) * stride / factor_escalado).astype(np.long)
        m_lowres = np.ceil(m.astype(float) * stride / factor_escalado).astype(np.long)
        w_lowres = np.ceil(stride/factor_escalado).astype(np.long)
        h_lowres = np.ceil(stride / factor_escalado).astype(np.long)
        tile_mask_lowres = clean_img_mask[m_lowres:m_lowres + h_lowres, n_lowres:n_lowres + w_lowres]
        if np.sum(tile_mask_lowres.flatten() == True) > 1:
            print((i * n_vector.size + j) / (m_vector.size* n_vector.size))
            sub_img = np.array(scan.read_region((n * stride, m * stride), 0, (tile_size, tile_size)), dtype=np.uint8)[
                      ..., 0:3]
            pil_im = Image.fromarray(sub_img, 'RGB')
            resized_im, seg_image = model.run(pil_im)
            aux = seg_image.astype(dtype=bool)
            aux = aux | wsi_seg[m * stride:m * stride + aux.shape[0], n * stride:n * stride + aux.shape[1]] # Eliminar zonas pequeñas
            aux = remove_small_objects(aux, min_size=1000)  # Eliminar zonas pequeñas
            aux = opening(aux, disk(10))  # Closing
            aux = binary_fill_holes(aux, structure=np.ones((10, 10)))  # Rellenar huecos
            wsi_seg[m * stride:m * stride + aux.shape[0], n * stride:n * stride + aux.shape[1]] = aux



#El resultado se almacena en JSON introduciendo recortes de la imagen (si no hay problemas de memoria)
##Numero de parches
crop_size = 5000
m_crops = np.fix(np.double(width)/crop_size)
n_crops = np.fix(np.double(height)/crop_size)

m_vector = np.arange(0, m_crops).astype(np.long)
n_vector = np.arange(0, n_crops).astype(np.long)

for i, m in enumerate(m_vector):
    for j, n in enumerate(n_vector):
        sub_wsi_seg = wsi_seg[m * crop_size: np.min([(m + 1) * crop_size, height]),
                      n * crop_size: np.min([(n + 1) * crop_size, height])]
        binaryImage2JSON("C:/Users/BlancaPT/PycharmProjects/BIGDATA/PI0032WEB/app/static/wsi/397W_HE_40x.json",
                         sub_wsi_seg.astype(dtype=np.uint8), x=n*crop_size, y=m*crop_size)
