import numpy as np
from PIL import Image
import tensorflow as tf
import openslide

from skimage.morphology import remove_small_objects, square, disk, binary_erosion, opening
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

    def runROI(self, filename, imgCoords):
        slide = openslide.OpenSlide(filename)

        cx = []
        cy = []

        for cxy in imgCoords:
            cx.append(float(cxy['x']))
            cy.append(float(cxy['y']))

        cx = np.asarray(cx)
        cy = np.asarray(cy)

        # Tamaño parches
        tile_size = self.INPUT_SIZE

        # Margen exterior

        ##Numero de parches

        m_tiles = 2
        n_tiles = 2

        # Stride
        dy = (np.max(cy) - np.min(cy)).astype(np.long)
        dx = (np.max(cx) - np.min(cx)).astype(np.long)

        # imagen que almacena segmentación
        print(dx, dy)
        wsi_seg = np.zeros((dy, dx), dtype=np.uint8)
        mask_seg = np.zeros((dy, dx), dtype=np.uint8)
        print(wsi_seg.shape)

        n_vector = np.linspace(0, dx - tile_size, num=n_tiles, dtype=np.long)
        m_vector = np.linspace(0, dy - tile_size, num=m_tiles, dtype=np.long)

        # Segmentacion
        for i, m in enumerate(m_vector):
            for j, n in enumerate(n_vector):
                print((i * n_vector.size + j) / (m_vector.size * n_vector.size))

                offset_x = np.array(int(slide.properties['openslide.bounds-x'])).astype(np.long)
                offset_y = np.array(int(slide.properties['openslide.bounds-y'])).astype(np.long)

                sub_img = np.array(slide.read_region((n + offset_x + np.min(cx).astype(np.long),
                                                      m + offset_y + np.min(cy).astype(np.long)),
                                                     0, (tile_size, tile_size)), dtype=np.uint8)[
                          ..., 0:3]
                pil_im = Image.fromarray(sub_img, 'RGB')
                resized_im, seg_image = self.run(pil_im)
                aux = seg_image.astype(dtype=bool)

                wsi_seg[m:m + aux.shape[0], n:n + aux.shape[1]] = \
                    aux.astype(np.uint8) + wsi_seg[m:m + aux.shape[0], n:n + aux.shape[1]]
                mask_seg[m:m + aux.shape[0], n:n + aux.shape[1]] = \
                    mask_seg[m:m + aux.shape[0], n:n + aux.shape[1]] + 1

                '''
                plt.subplot(121)
                plt.imshow(aux)
                plt.subplot(122)
                plt.imshow(sub_img)
                plt.show()
                '''

        # plt.subplot(231)
        # plt.imshow(wsi_seg)

        # mask_seg[mask_seg >= 2] = 2
        wsi_seg = (wsi_seg > 0) * wsi_seg
        wsi_seg = wsi_seg.astype(np.bool)
        wsi_seg = opening(wsi_seg, disk(3))
        wsi_seg = remove_small_objects(wsi_seg.copy(), min_size=500, connectivity=1)  # Eliminar zonas pequeñas (FG)
        # plt.subplot(232)
        # plt.imshow(wsi_seg)
        bg = np.invert(remove_small_objects(np.invert(wsi_seg.copy()), min_size=500,
                                            connectivity=1))  # Eliminar zonas pequeñas (BG)
        wsi_seg = wsi_seg | bg

        # plt.subplot(233)
        # plt.imshow(wsi_seg)
        # plt.show()
        contours, hierarchy = cv2.findContours(wsi_seg.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        data = []

        aux = {"applyMatrix": True,
               "segments": [],
               "closed": True,
               "fillColor": [0.61961, 0.61961, 0.61961, 0.5],
               "strokeColor": [0, 0, 0],
               "strokeScaling": False}
        for i, contour in enumerate(contours):
            xv = contour[:, 0, 0]
            yv = contour[:, 0, 1]
            data.append({
                'imgCoords': [
                    {
                        "x": float(xi + np.min(cx)),
                        "y": float(yi + np.min(cy))
                    }
                    for xi, yi in zip(xv, yv)],
                "path": ["Path", aux],
                "zoom": "0.016597"})
        return data


if __name__ == '__main__':
    slide = openslide.OpenSlide('/media/telemed/BDPATH_WEBIM/PI0032WEB/P005-HE-047-6.mrxs')
    print(np.array(int(slide.properties['openslide.bounds-x'])).astype(np.long))
