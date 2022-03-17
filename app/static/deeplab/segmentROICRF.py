
from scipy.misc import imsave, imresize
import tensorflow as tf
from skimage.morphology import remove_small_objects, square, disk, binary_erosion, opening, ball
from skimage.transform import resize
from PIL import Image
import openslide
import numpy as np
import matplotlib.pyplot as plt
import cv2

from pydensecrf.utils import compute_unary, create_pairwise_bilateral, \
    create_pairwise_gaussian, softmax_to_unary
import pydensecrf.densecrf as dcrf

class DeepLabModel(object):
    """Class to load deeplab model and run inference."""
    INPUT_TENSOR_NAME = 'ImageTensor:0'
    OUTPUT_TENSOR_NAME = 'ResizeBilinear_1: 0'
    INPUT_SIZE = 500

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

        seg_map = resize(seg_map, (width, height, seg_map.shape[2]))
        seg_map = seg_map[:, :, 0:2]
        max_ = np.max(seg_map, axis=2)
        max_ = max_[:, :, np.newaxis]

        e_seg_map = np.exp(seg_map - max_)
        e_seg_map_sum = e_seg_map.sum(axis=2)
        e_seg_map_sum = e_seg_map_sum[:, :, np.newaxis]

        e_seg_map = np.divide(e_seg_map, e_seg_map_sum)  # only difference

        return resized_image, e_seg_map

    def runROI(self, filename, imgCoords):
        slide = openslide.OpenSlide(filename)

        cx = []
        cy = []

        for cxy in imgCoords:
            cx.append(float(cxy['x']))
            cy.append(float(cxy['y']))

        cx = np.asarray(cx)
        cy = np.asarray(cy)

        tile_size = self.INPUT_SIZE

        # Stride
        dy = (np.max(cy) - np.min(cy)).astype(np.long)
        dx = (np.max(cx) - np.min(cx)).astype(np.long)

        m_tiles = 3
        n_tiles = 3

        wsi_seg = np.zeros((dy, dx, 2), dtype=float)
        mask_seg = np.zeros((dy, dx, 1), dtype=float)
        img = np.zeros((dy, dx, 3), dtype=np.uint8)

        n_vector = np.linspace(0, dx - tile_size, num=n_tiles, dtype=np.long)
        m_vector = np.linspace(0, dy - tile_size, num=m_tiles, dtype=np.long)



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

                wsi_seg[m:m + seg_image.shape[0], n:n + seg_image.shape[1], :] = \
                    seg_image + wsi_seg[m:m + seg_image.shape[0], n:n + seg_image.shape[1], :]
                mask_seg[m:m + seg_image.shape[0], n:n + seg_image.shape[1], 0] = \
                    mask_seg[m:m + seg_image.shape[0], n:n + seg_image.shape[1], 0] + 1
                img[m:m + seg_image.shape[0], n:n + seg_image.shape[1], :] = sub_img

        wsi_seg = np.divide(wsi_seg, mask_seg)
        unary = softmax_to_unary(np.reshape(wsi_seg, (wsi_seg.shape[0] * wsi_seg.shape[1], 2)))
        unary = np.ascontiguousarray(unary.transpose())
        d = dcrf.DenseCRF(wsi_seg.shape[0] * wsi_seg.shape[1], 2)

        d.setUnaryEnergy(unary)
        feats = create_pairwise_gaussian(sdims=(5, 5), shape=wsi_seg.shape[:2])

        d.addPairwiseEnergy(feats, compat=3,
                            kernel=dcrf.DIAG_KERNEL,
                            normalization=dcrf.NORMALIZE_SYMMETRIC)
        Q = d.inference(5)

        res = np.argmax(Q, axis=0).reshape((wsi_seg.shape[0], wsi_seg.shape[1]))

        '''
        plt.subplot(131)
        plt.imshow(img)
        plt.subplot(132)
        plt.imshow(wsi_seg[:, :, 0])
        plt.subplot(133)
        plt.imshow(res)
        plt.show()
        '''

        contours, hierarchy = cv2.findContours(res.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
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



if __name__=='__main__':
    slide = openslide.OpenSlide('/media/telemed/BDPATH_WEBIM/PI0032WEB/P005-HE-047-6.mrxs')
    print(np.array(int(slide.properties['openslide.bounds-x'])).astype(np.long))

