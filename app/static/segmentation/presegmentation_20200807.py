import numpy as np
import openslide
import random
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import watershed
from skimage import color
from skimage.filters import sobel
from skimage.color import rgb2hed
from skimage.morphology import disk, square
from skimage.filters.rank import gradient
from skimage.morphology import remove_small_objects, remove_small_holes, thin, binary_erosion, opening, erosion, \
    dilation, binary_dilation, binary_closing, binary_opening
from skimage.measure import label, regionprops, find_contours
from skimage.filters import gaussian, sobel
from skimage.filters import rank
from skimage.transform import rescale
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage.morphology import binary_fill_holes
from scipy import linalg
import scipy.stats as stats
from skimage.segmentation import active_contour
from skimage import exposure
from skimage.exposure import rescale_intensity

from sklearn.preprocessing import StandardScaler
from skimage.color import label2rgb
from PIL import Image

debug = 0

from sklearn.mixture import GaussianMixture


def getScaleFactor(scan, level_inicial, level_final):
    dimi = scan.level_dimensions[level_inicial]
    dimf = scan.level_dimensions[level_final]
    factor_x = np.round(float(dimf[0]) / float(dimi[0]))
    factor_y = float(dimf[1]) / float(dimi[1])
    return [factor_x, factor_y]


def normImage(im, max_=[], min_=[]):
    if max_:
        if max_ - min_ > 0:
            im = (im - min_) / (max_ - min_)
    else:
        max_ = np.max(im)
        min_ = np.min(im)
        if max_ - min_ > 0:
            im = (im - min_) / (max_ - min_)
    return im


def postcontours(c):
    xv = c[:, 1]
    yv = c[:, 0]
    if xv[-1] == xv[0] and yv[-1] == yv[0]:
        xv = xv[:-1]
        yv = yv[:-1]
    nx = []
    ny = []
    for i, (x, y) in enumerate(zip(xv, yv)):
        nx.append(x)
        ny.append(y)
        p2 = [x, y]
        if i == 0:
            p1 = [xv[-1], yv[-1]]
        else:
            p1 = [xv[i - 1], yv[i - 1]]
        if i >= xv.size - 2:
            if i == xv.size - 2:
                p3 = [xv[i + 1], yv[i + 1]]
            elif i == xv.size - 1:
                p3 = [xv[0], yv[0]]
        else:
            p3 = [xv[i + 1], yv[i + 1]]

        # Linea post
        dxl = p3[0] - p2[0]
        dyl = p3[1] - p2[1]
        # Linea prev
        dxlprev = p2[0] - p1[0]

        if dxl != 0 and dyl != 0:
            if dxlprev == 0:
                ny.append(p3[1])
                nx.append(p2[0])
            else:
                ny.append(p2[1])
                nx.append(p3[0])

    newc = np.zeros((len(nx), 2))
    newc[:, 1] = np.asarray(nx)
    newc[:, 0] = np.asarray(ny)
    newc = np.vstack((newc, newc[0, :]))
    return newc


def extractMainTissue(im, staining_type="IMM"):
    maskIm = np.invert(np.logical_and(np.logical_and(im[:, :, 0] == 0, im[:, :, 1] == 0),
                                      im[:, :, 2] == 0))
    labelIm = label(maskIm)
    rp = regionprops(labelIm)
    centroids = [region.centroid for region in rp]  # Tuple: (row, col)
    areas = [region.area for region in rp]  # Tuple: (row, col)

    if staining_type == "HE":
        # Filtrar por área
        i_big_enough = np.argwhere(areas > np.max(areas) * 0.3)

        i_big_enough = np.squeeze(i_big_enough)
        imgBG = np.zeros(maskIm.shape, dtype=np.bool)
        if i_big_enough.size == 1:
            imgBG[rp[i_big_enough].coords[:, 0], rp[i_big_enough].coords[:, 1]] = True
        else:
            for ibe in i_big_enough:
                imgBG[rp[ibe].coords[:, 0], rp[ibe].coords[:, 1]] = True
        lowResFG = np.copy(im[:, :, 0:3]) * imgBG[:, :, np.newaxis].astype(dtype=np.float)

        contours, hierarchy = cv2.findContours(imgBG.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    else:
        # Filtrar por posición y area
        distances = []  # Distancia al centro
        for ibe, d in enumerate(centroids):
            imgBG = np.zeros(maskIm.shape, dtype=np.bool)
            imgBG[rp[ibe].coords[:, 0], rp[ibe].coords[:, 1]] = True
            distances.append(im.shape[0] - d[0])
        distances = np.asarray(distances)
        areas = np.asarray(areas)
        i_big_enough_up = np.argwhere(np.bitwise_and(areas > np.max(areas) * 0.2, distances > np.min(distances)))
        i_big_enough_up = np.squeeze(i_big_enough_up)
        imgBG = np.zeros(maskIm.shape, dtype=np.bool)
        if areas.size == 1:  # Solo una zona
            imgBG[rp[0].coords[:, 0], rp[0].coords[:, 1]] = True

        else:
            if i_big_enough_up.size == 1:  # Solo una zona tras filtrar
                imgBG[rp[i_big_enough_up].coords[:, 0], rp[i_big_enough_up].coords[:, 1]] = True

            else:
                for ibe in i_big_enough_up:  # Mas de una zona
                    imgBG[rp[ibe].coords[:, 0], rp[ibe].coords[:, 1]] = True
        lowResFG = np.copy(im[:, :, 0:3]) * imgBG[:, :, np.newaxis].astype(dtype=np.float)

        contours, hierarchy = cv2.findContours(imgBG.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]

    return lowResFG, contours


def normHema(im):
    im = (im + 0.6950) / (-0.2467 + 0.6950)
    return im


def normDab(im):
    im = (im + 0.5426) / (-0.1437 + 0.5426)
    return im


def getHemaDabNormParam(slide, contours, level_lowRes, level_highRes):
    factor_lowtomaxres = getScaleFactor(slide, level_lowRes, 0)

    cxmin = np.min(contours[0][:, 0, 0])
    cymin = np.min(contours[0][:, 0, 1])
    cxmax = np.max(contours[0][:, 0, 0])
    cymax = np.max(contours[0][:, 0, 1])

    factor_lowtohighres = getScaleFactor(slide, level_lowRes, level_highRes)
    dx = (cxmax - cxmin) * factor_lowtohighres[0]
    dy = (cymax - cymin) * factor_lowtohighres[1]

    clim = np.asarray([cxmin * factor_lowtomaxres[0], cymin * factor_lowtomaxres[1],
                       cxmax * factor_lowtomaxres[0], cymax * factor_lowtomaxres[1]]).astype(np.long)

    max_dim = 500

    x_vector = np.arange(0, np.ceil(dx / max_dim) + 1) * max_dim
    y_vector = np.arange(0, np.ceil(dy / max_dim) + 1) * max_dim

    x_vector[x_vector > dx] = dx
    y_vector[y_vector > dy] = dy

    wx = int(x_vector[1] - x_vector[0])
    wy = int(y_vector[1] - y_vector[0])

    factor_hightomaxres = getScaleFactor(slide, level_highRes, 0)
    x_vector = x_vector * factor_hightomaxres[0]
    y_vector = y_vector * factor_hightomaxres[1]

    x_vector = x_vector.astype(dtype=np.int32)
    y_vector = y_vector.astype(dtype=np.int32)

    N_ROI = 100
    count = 0
    hema_values = []
    dab_values = []

    while count < N_ROI:

        y = y_vector[random.randint(0, y_vector.size - 1)]
        x = x_vector[random.randint(0, x_vector.size - 1)]
        cx = int(clim[0] + x)
        cy = int(clim[1] + y)
        im = color.rgba2rgb(np.asarray(slide.read_region((cx, cy), level_highRes, (int(wx / 2), int(wy / 2))),
                                       dtype=np.float32) / 255)
        # Eliminar zonas blancas
        if np.sum(im[:, :, 0].flatten() > 0.9) < im.shape[0] * im.shape[1] * 0.80:

            print(count / N_ROI)
            ihc_hed = rgb2hed(im)

            dab = normDab(ihc_hed[:, :, 2])
            hema = normHema(ihc_hed[:, :, 0])

            selection_element = disk(3)  # matrix of n pixels with a disk shape

            dab_blur = gaussian(dab, sigma=1.5)
            aux1 = dab_blur.copy()
            aux1[dab_blur < 0.4] = 0
            im_binary1 = aux1 > 0
            im_binary1 = remove_small_objects(im_binary1, min_size=400)
            im_binary1 = binary_erosion(im_binary1, selection_element)
            dab_values.append(dab_blur[im_binary1].flatten())

            hema_blur = gaussian(hema, sigma=1.5)
            aux2 = hema_blur.copy()
            aux2[hema_blur < 0.4] = 0
            im_binary2 = aux2 > 0
            im_binary2 = remove_small_objects(im_binary2, min_size=400)
            im_binary2 = binary_erosion(im_binary2, selection_element)
            hema_values.append(hema_blur[im_binary2].flatten())
            if (np.sum(im_binary2) > 0):
                count = count + 1
            if 0:
                plt.subplot(241)
                plt.imshow(im)
                plt.subplot(242)
                plt.imshow(dab)
                plt.subplot(243)
                plt.imshow(aux1)
                plt.subplot(244)
                plt.imshow(im_binary1)
                plt.subplot(246)
                plt.imshow(hema)
                plt.subplot(247)
                plt.imshow(aux2)
                plt.subplot(248)
                plt.imshow(im_binary2)
                plt.show()

    hema_values = np.asarray(np.hstack(hema_values))
    hema_values = hema_values[hema_values > 0]
    dab_values = np.asarray(np.hstack(dab_values))

    mean_hema = np.mean(hema_values)
    std_hema = np.std(hema_values)
    mean_dab = np.mean(dab_values)
    std_dab = np.std(dab_values)
    if debug:
        plt.subplot(121)
        plt.hist(hema_values, 100)
        plt.title(mean_hema - std_hema)
        plt.subplot(122)
        plt.hist(dab_values, 100)
        plt.title(mean_dab - std_dab)
        plt.show()
    return mean_hema, std_hema, mean_dab, std_dab


def getHemaDabNormParamROI(im, mean_hema, std_hema, mean_dab, std_dab):
    ihc_hed = rgb2hed(im)

    dab = opening(normDab(ihc_hed[:, :, 2]), disk(7))
    hema = opening(normHema(ihc_hed[:, :, 0]), disk(7))

    selection_element = disk(3)  # matrix of n pixels with a disk shape

    hema_values = []
    dab_values = []

    dab_blur = gaussian(dab, sigma=1.5)
    aux1 = dab_blur.copy()
    aux1[dab_blur < 0.4] = 0
    im_binary1 = aux1 > 0
    im_binary1 = remove_small_objects(im_binary1, min_size=400)
    im_binary1 = binary_erosion(im_binary1, selection_element)
    dab_values.append(dab_blur[im_binary1].flatten())

    hema_blur = gaussian(hema, sigma=1.5)
    aux2 = hema_blur.copy()
    aux2[hema_blur < 0.35] = 0
    im_binary2 = aux2 > 0
    im_binary2 = remove_small_objects(im_binary2, min_size=400)
    im_binary2 = binary_erosion(im_binary2, selection_element)
    hema_values.append(hema_blur[im_binary2].flatten())

    if 0:
        plt.subplot(241)
        plt.imshow(im)
        plt.subplot(242)
        plt.imshow(dab)

        plt.subplot(243)
        plt.imshow(aux1)
        plt.subplot(244)
        plt.imshow(im_binary1)
        plt.subplot(246)
        plt.imshow(hema)
        plt.title(str(mean_hema) + " " + str(std_hema))
        plt.subplot(247)
        plt.imshow(aux2)
        plt.subplot(248)
        plt.imshow(im_binary2)
        plt.show()

    hema_values = np.asarray(np.hstack(hema_values))
    hema_values = hema_values[hema_values > 0]
    dab_values = np.asarray(np.hstack(dab_values))

    if 0:
        plt.subplot(121)
        plt.hist(hema_values, 100)
        plt.title("HEMA: " + str(mean_hema) + " " + str(np.mean(hema_values)))
        plt.subplot(122)
        plt.hist(dab_values, 100)
        plt.title("DAB: " + str(mean_dab) + " " + str(np.mean(dab_values)))
        plt.show()

    mean_hema_n = np.mean(hema_values)
    std_hema_n = np.std(hema_values)
    mean_dab_n = np.mean(dab_values)
    std_dab_n = np.std(dab_values)

    return mean_hema_n, std_hema_n, mean_dab_n, std_dab_n


def nuclearBioMarkerROI(im, th):
    im_binary = im > th

    regions = regionprops(label(im_binary))

    area = [region.area for region in regions]
    if debug:
        plt.subplot(1, 3, 1)
        plt.imshow(im)
        plt.subplot(1, 3, 2)
        plt.imshow(im_binary)
        plt.subplot(1, 3, 3)
        plt.stem(np.sort(area))
        plt.show()

    area_m = np.mean(area)

    # diameter = int(np.sqrt(area_m/np.pi)/2)
    diameter = 7
    im_aux = im.copy()
    im_aux[im_binary == 0] = 0
    im_aux[im_aux > 1] = 1

    # distance = normImage(ndi.distance_transform_edt(im_aux)) + normImage(sobel(im_aux))
    distance = normImage(gradient(im_aux, disk(int(diameter / 2))))
    distance = 1 - normImage(distance)
    distance = np.multiply(distance, im_binary.astype(np.float64))

    local_maxi = peak_local_max(distance, indices=False, min_distance=diameter, threshold_abs=0.9,
                                labels=im_binary)
    local_maxi = remove_small_objects(local_maxi, min_size=20)
    distance = np.multiply(distance, local_maxi.astype(np.float64))
    markers = ndi.label(local_maxi)[0]
    if debug:
        plt.subplot(151)
        plt.imshow(im)
        plt.subplot(152)
        plt.imshow(im_aux)
        plt.subplot(153)
        plt.imshow(distance)
        plt.subplot(154)
        plt.imshow(markers)
        plt.show()

    labels_watershed = watershed(-distance, markers, mask=im_binary)
    # labels_watershed= label(distance>0)

    im_holes = remove_small_holes(im_binary, area_threshold=2000, connectivity=1, in_place=False)
    im_smallremoved = remove_small_objects(im_holes, min_size=100)
    im_label_filtered = label(im_smallremoved)

    if debug:
        plt.subplot(131)
        plt.imshow(im)
        plt.subplot(132)
        plt.imshow(im_label_filtered)
        plt.subplot(133)
        plt.imshow(labels_watershed)
        plt.show()

    return im_label_filtered, labels_watershed


def nuclearBioMarkerHotSpot(filename, imgCoords):
    slide = openslide.OpenSlide(filename)
    level_lowRes = slide.level_count - 3
    level_highRes = level_lowRes - 8

    lowRes = np.asarray(slide.read_region((0, 0), level_lowRes,
                                          slide.level_dimensions[level_lowRes]),
                        dtype=np.float32) / 255

    mainTissue, contours = extractMainTissue(lowRes)

    mean_hema, std_hema, mean_dab, std_dab = getHemaDabNormParam(slide, contours, level_lowRes, level_highRes)

    cx = []
    cy = []
    print(imgCoords)
    for cxy in imgCoords:
        cx.append(float(cxy['x']))
        cy.append(float(cxy['y']))

    cx = np.asarray(cx)
    cy = np.asarray(cy)

    offset_x = np.array(int(slide.properties['openslide.bounds-x'])).astype(np.long)
    offset_y = np.array(int(slide.properties['openslide.bounds-y'])).astype(np.long)

    tile_size = (np.max(cx) - np.min(cx)).astype(np.long)
    im = color.rgba2rgb(np.array(slide.read_region((offset_x + np.min(cx).astype(np.long),
                                                    offset_y + np.min(cy).astype(np.long)),
                                                   0, (tile_size, tile_size)), dtype=np.float32) / 255)

    mean_hema, std_hema, trash_, trash_ = getHemaDabNormParamROI(im, mean_hema, std_hema, mean_dab, std_dab)

    data = []
    ihc_hed = rgb2hed(im)
    hema = opening(normHema(ihc_hed[:, :, 0]), disk(7))
    dab = opening(normDab(ihc_hed[:, :, 2]), disk(7))

    hema_nucleus, hema_segmentation = nuclearBioMarkerROI(hema, mean_hema - std_hema)

    dab_nucleus, dab_segmentation = nuclearBioMarkerROI(dab, mean_dab - std_dab)

    hema_segmentation = (hema_segmentation).astype(np.uint8) * 255

    labels_hema_segmentation = np.unique(hema_segmentation.flatten())
    labels_hema_segmentation = labels_hema_segmentation[labels_hema_segmentation > 0]

    contours = []
    dab_segmentation_binary = dab_segmentation.copy() > 0
    for l in labels_hema_segmentation:
        aux = np.zeros(hema_segmentation.shape)
        aux[hema_segmentation == l] = 1
        hema_dab_intersection = np.bitwise_and(dab_segmentation_binary, aux.astype(dtype=np.bool))
        n = np.sum(hema_dab_intersection.flatten())
        if n == 0:
            contours_, hierarchy = cv2.findContours(aux.astype(np.uint8) * 255, cv2.RETR_TREE,
                                                    cv2.CHAIN_APPROX_SIMPLE)[-2:]
            contours = contours + contours_
    aux = {"applyMatrix": True,
           "segments": [],
           "closed": True,
           "strokeColor": [0, 0, 0],
           "fillColor": [0.2, 0.51, 168 / 155, 0.5],
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
            "zoom": "0.016597",
            "name": "Negative Nuclei"})
    n_c = len(contours)

    dab_segmentation = (dab_segmentation).astype(np.uint8) * 255

    labels_dab_segmentation = np.unique(dab_segmentation.flatten())
    labels_dab_segmentation = labels_dab_segmentation[labels_dab_segmentation > 0]

    contours = []
    for l in labels_dab_segmentation:
        aux = np.zeros(dab_segmentation.shape)
        aux[dab_segmentation == l] = 1
        contours_, hierarchy = cv2.findContours(aux.astype(np.uint8) * 255, cv2.RETR_TREE,
                                                cv2.CHAIN_APPROX_SIMPLE)[-2:]
        contours = contours + contours_

    aux = {"applyMatrix": True,
           "segments": [],
           "closed": True,
           "strokeColor": [0, 0, 0],
           "fillColor": [210 / 255, 0, 0, 0.5],
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
            "zoom": "0.016597",
            "name": "Positive Nuclei"})
    p_c = len(contours)
    if p_c + n_c == 0:
        score = 0
    else:
        score = p_c / (p_c + n_c)
    return data, score


def nuclearBioMarkerSmallPiece(slide, biomarker_type, contours, max_dim, level_lowRes, level_highRes, mean_hema,
                               std_hema, mean_dab, std_dab):
    factor_lowtomaxres = getScaleFactor(slide, level_lowRes, 0)

    cxmin = np.min(contours[:, 0, 0])
    cymin = np.min(contours[:, 0, 1])
    cxmax = np.max(contours[:, 0, 0])
    cymax = np.max(contours[:, 0, 1])

    factor_lowtohighres = getScaleFactor(slide, level_lowRes, level_highRes)
    dx = (cxmax - cxmin) * factor_lowtohighres[0]
    dy = (cymax - cymin) * factor_lowtohighres[1]

    clim = np.asarray([cxmin * factor_lowtomaxres[0], cymin * factor_lowtomaxres[1],
                       cxmax * factor_lowtomaxres[0], cymax * factor_lowtomaxres[1]]).astype(np.long)

    x_vector = np.arange(0, np.ceil(dx / max_dim) + 1) * max_dim
    y_vector = np.arange(0, np.ceil(dy / max_dim) + 1) * max_dim

    x_vector[x_vector > dx] = dx
    y_vector[y_vector > dy] = dy

    wx = int(x_vector[1] - x_vector[0])
    wy = int(y_vector[1] - y_vector[0])

    factor_hightomaxres = getScaleFactor(slide, level_highRes, 0)
    x_vector = x_vector * factor_hightomaxres[0]
    y_vector = y_vector * factor_hightomaxres[1]

    x_vector = x_vector.astype(dtype=np.int32)
    y_vector = y_vector.astype(dtype=np.int32)

    im_negative = np.zeros((y_vector.size * 2, x_vector.size * 2))
    im_positive = np.zeros((y_vector.size * 2, x_vector.size * 2))

    for py, y in enumerate(y_vector[:-1]):
        print(py, '/', len(y_vector[:-1]))
        for px, x in enumerate(x_vector[:-1]):
            # if py<25:
            cx = int(clim[0] + x)
            cy = int(clim[1] + y)

            im = color.rgba2rgb(np.asarray(slide.read_region((cx, cy), level_highRes, (wx, wy)),
                                           dtype=np.float32) / 255)
            if np.sum(im[:, :, 0].flatten() > 0.9) < im.shape[0] * im.shape[1] * 0.80:
                ihc_hed = rgb2hed(im)
                dab = normDab(ihc_hed[:, :, 2])
                hema = normHema(ihc_hed[:, :, 0])

                selection_element = disk(3)  # matrix of n pixels with a disk shape

                dab_blur = gaussian(dab, sigma=1.5)
                aux1 = dab_blur.copy()
                aux1[dab_blur < mean_dab - std_dab] = 0
                im_binary1 = aux1 > 0
                im_binary1 = remove_small_objects(im_binary1, min_size=200)
                im_positive_aux = binary_erosion(im_binary1, selection_element)

                hema_blur = gaussian(hema, sigma=1.5)
                aux2 = hema_blur.copy()
                aux2[hema_blur < mean_hema - std_hema] = 0
                im_binary2 = aux2 > 0
                im_binary2 = remove_small_objects(im_binary2, min_size=200)
                im_negative_aux = binary_erosion(im_binary2, selection_element)

                sum_positive = np.sum(im_positive_aux.flatten())
                sum_negative = np.sum(im_negative_aux.flatten())
                if debug:
                    plt.subplot(231)
                    plt.imshow(im)
                    plt.subplot(232)
                    plt.imshow(hema_blur)
                    plt.title(mean_hema - std_hema)
                    plt.subplot(233)
                    plt.imshow(dab_blur)
                    plt.title(mean_dab - std_dab)
                    plt.subplot(235)
                    plt.imshow(im_binary2)
                    plt.subplot(236)
                    plt.imshow(im_binary1)
                    plt.show()
                if sum_negative > sum_positive or biomarker_type == "ER":
                    im_positive[py * 2:py * 2 + 2, px * 2:px * 2 + 2] = sum_positive
                else:
                    im_positive[py * 2:py * 2 + 2, px * 2:px * 2 + 2] = 0

                im_negative[py * 2:py * 2 + 2, px * 2:px * 2 + 2] = sum_negative

    return im_positive, im_negative, clim;


def nuclearBioMarker(filename, biomarker_type):
    # Funcion colormap general

    slide = openslide.OpenSlide(filename)
    level_lowRes = slide.level_count - 3
    level_highRes = level_lowRes - 8

    lowRes = np.asarray(slide.read_region((0, 0), level_lowRes,
                                          slide.level_dimensions[level_lowRes]),
                        dtype=np.float32) / 255

    offset_x = int(slide.properties['openslide.bounds-x'])
    offset_y = int(slide.properties['openslide.bounds-y'])

    mainTissue, contours_mainTissue = extractMainTissue(lowRes)
    if debug:
        plt.subplot(121)
        plt.imshow(lowRes)
        plt.subplot(122)
        plt.imshow(mainTissue)
        plt.show()

    data = []
    mean_hema, std_hema, mean_dab, std_dab = \
        getHemaDabNormParam(slide, contours_mainTissue, level_lowRes, level_highRes)

    im_positive = []
    im_negative = []
    clim = []
    max_dim = 500
    for c in contours_mainTissue:
        im_positive_aux, im_negative_aux, clim_aux = nuclearBioMarkerSmallPiece(slide, biomarker_type, c, max_dim,
                                                                                level_lowRes, level_highRes,
                                                                                mean_hema, std_hema, mean_dab, std_dab)
        im_positive.append(im_positive_aux)
        im_negative.append(im_negative_aux)
        clim.append(clim_aux)
        if debug:
            plt.subplot(131)
            plt.imshow(im_negative_aux)
            plt.subplot(132)
            plt.imshow(im_positive_aux)
            plt.subplot(133)
            plt.stem(np.sort(im_positive_aux[im_positive_aux > 0].flatten()))
            plt.show()

    q = np.linspace(0, 1, 10)

    all_pos = np.array([], dtype=np.float64)

    for im_pos in im_positive:
        all_pos = np.hstack((all_pos, im_pos[im_pos > 0].flatten()))

    perc_th = np.percentile(all_pos, 95)
    divisor = []
    im_ratio = []

    for i, im_pos in enumerate(im_positive):
        im_positive[i][im_pos > perc_th] = perc_th
        divisor.append(im_positive[i] + im_negative[i])
        divisor[i][divisor[i] == 0] = 1
        im_ratio.append(im_positive[i] / divisor[i])
    all_pos = np.array([], dtype=np.float64)
    for im_pos in im_positive:
        all_pos = np.hstack((all_pos, im_pos.flatten()))
    min_pos = np.min(all_pos)
    max_pos = np.max(all_pos)

    all_ratio = np.array([], dtype=np.float64)
    for im_rat in im_ratio:
        all_ratio = np.hstack((all_ratio, im_rat.flatten()))

    min_ratio = np.min(all_ratio)
    max_ratio = np.max(all_ratio)

    for i, im_pos in enumerate(im_positive):
        im_positive[i] = normImage(im_pos, max_=max_pos, min_=min_pos)

    for i, im_rat in enumerate(im_ratio):
        im_ratio[i] = normImage(im_rat, max_=max_ratio, min_=min_ratio)

    im_ratio = im_positive

    # para evitar regiones con pocos nucleos
    all_neg = np.array([], dtype=np.float64)

    for im_neg in im_negative:
        all_neg = np.hstack((all_neg, im_neg[im_neg > 0].flatten()))

    perc_th = np.percentile(all_neg, 50)

    for i, im_rat in enumerate(im_ratio):
        if biomarker_type != "ER":
            im_ratio[i][im_negative[i] < perc_th] = 0  # zonas no lo suficientemente negativas
            im_ratio[i][im_positive[i] == 0] = 0  # zonas no positivas
        # Promediado
        im_ratio[i] = rank.mean(im_ratio[i], selem=disk(2))
        # Rescalado de dimension para hacer tiles del tamaño original
        im_ratio[i] = rescale(np.squeeze(im_ratio[i]), 0.25)
        im_ratio[i] = rescale(np.squeeze(im_ratio[i]), 4, order=0)
        if debug:
            plt.subplot(131)
            plt.imshow(im_ratio[i])

            plt.show()

    contours = []
    coordenadas_x = []
    coordenadas_y = []
    opacity = []

    factor_hightomaxres = getScaleFactor(slide, level_highRes, 0)
    for ic, im_colormap in enumerate(im_ratio):
        for i, q_ in enumerate(q[:-1]):
            im_colormap[np.logical_and(im_colormap > q[i], im_colormap <= q[i + 1])] = q[i + 1]
            im_binary = np.zeros(im_colormap.shape)
            im_binary[np.logical_and(im_colormap > q[i], im_colormap <= q[i + 1])] = 1

            out = label(im_binary, connectivity=1)
            contour = find_contours(im_binary, level=0.5, fully_connected='low', positive_orientation="high")

            for kk, c in enumerate(contour):
                c = postcontours(c)
                contours.append(c)
                xv = (clim[ic][0] + (c[:, 1] / 2 * max_dim) * factor_hightomaxres[0] - offset_x).astype(dtype=np.uint)
                yv = (clim[ic][1] + (c[:, 0] / 2 * max_dim) * factor_hightomaxres[1] - offset_y).astype(dtype=np.uint)
                coordenadas_x.append(xv)
                coordenadas_y.append(yv)
                opacity.append(q[i + 1])

    opacity_array = np.asarray(opacity)

    min_ = np.min(opacity_array)
    max_ = np.max(opacity_array)

    if max_ == min_:
        min_ = 0
    for (xv, yv, opa) in zip(coordenadas_x, coordenadas_y, opacity):
        aux = {"applyMatrix": True,
               "segments": [],
               "closed": True,
               "fillColor": [245 / 255., 206 / 255., 66 / 255., (opa - min_) / (max_ - min_) * 0.6 + 0.2],
               "strokeWidth": 0,
               "strokeColor": [0, 0, 0],
               "strokeAlpha": 0,
               "strokeScaling": False}
        data.append({
            'hemagCoords': [
                {
                    "x": float(xi),
                    "y": float(yi)
                }
                for xi, yi in zip(xv, yv)],
            "path": ["Path", aux],
            "zoom": "0.016597",
            "name": biomarker_type + " Colormap"})

    return data


def membraneHemaBioMarkerROI(im, th):
    im_binary = im > th
    diameter = 7
    im_aux = im.copy()
    im_aux[im_binary == 0] = 0

    # distance = normImage(ndi.distance_transform_edt(im_aux)) + normImage(sobel(im_aux))
    distance = normImage(gradient(im_aux, disk(int(diameter / 2))))
    distance = 1 - normImage(distance)
    distance = np.multiply(distance, im_binary.astype(np.float64))

    local_maxi = peak_local_max(distance, indices=False, min_distance=diameter, threshold_abs=0.9,
                                labels=im_binary)
    local_maxi = remove_small_objects(local_maxi, min_size=20)
    distance = np.multiply(distance, local_maxi.astype(np.float64))
    markers = ndi.label(local_maxi)[0]
    if debug:
        plt.subplot(151)
        plt.imshow(im)
        plt.subplot(152)
        plt.imshow(im_aux)
        plt.subplot(153)
        plt.imshow(distance)
        plt.subplot(154)
        plt.imshow(markers)
        plt.show()

    labels_watershed = watershed(-distance, markers, mask=im_binary)

    return labels_watershed


def membraneDabBioMarkerROI_old(hema, dab, th):
    im_location = np.zeros(hema.shape, dtype=np.bool)
    steps = np.linspace(0, 3, 10)
    area_steps = np.linspace(1000, 10000, 10)[::-1]
    print(area_steps)
    for i, f in enumerate(steps):
        dab_mod = dab - hema * f
        dab_mod[dab_mod < 0] = 0
        dab_mod = (dab_mod * 255).astype(np.uint8)
        dab_mod_pil = Image.fromarray(dab_mod)
        dab_mod_pil_q = dab_mod_pil.quantize(colors=8, method=2, dither=0)
        dab_mod_pil_q = np.asarray(dab_mod_pil_q)

        (values, counts) = np.unique(dab_mod_pil_q[dab_mod < th * 255], return_counts=True)
        l = values[np.argmax(counts)]

        binary = remove_small_objects(dab_mod_pil_q == l, min_size=100)
        binary = remove_small_holes(binary, area_threshold=100, connectivity=1, in_place=False)
        labelIm = label(binary)
        rp = regionprops(labelIm)
        for region in rp:
            if region.area < 5000 and region.area > 200:
                im_location[region.coords[:, 0], region.coords[:, 1]] = True

    im_location = remove_small_holes(im_location, area_threshold=2000, connectivity=1, in_place=False)
    if debug:
        plt.subplot(121)
        plt.imshow(dab)
        plt.subplot(122)
        plt.imshow(im_location + dab)
        plt.show()

    return im_location


def membraneDabBioMarkerROI(im, dab, mask):
    nivel = 3
    dab_preseg = dab.copy()
    mean_dab = np.mean(dab[dab > 0].flatten())
    std_dab = np.std(dab[dab > 0].flatten())
    perimeter = np.bitwise_xor(binary_dilation(mask.copy(), disk(3)), mask)
    im_location = np.zeros(dab.shape, dtype=np.bool)

    for i in np.arange(0, nivel):
        dab_preseg[dab_preseg <= mean_dab - std_dab / 2] = 0
        mean_dab = np.mean(dab_preseg[dab_preseg > 0].flatten())
        std_dab = np.std(dab[dab > 0].flatten())
        binary_holes = (dab_preseg > 0)
        aux1 = binary_holes.copy()
        binary_holes = remove_small_objects(binary_holes, min_size=400)
        aux2 = binary_holes.copy()
        # binary_holes = (dab_preseg > 0).astype(dtype=np.int) | perimeter
        binary_holes = binary_closing(binary_holes, disk(10))
        binary_holes = ~binary_holes
        binary_holes[~binary_erosion(mask, disk(10))] = 0
        rp = regionprops(label(binary_holes))
        for region in rp:
            if region.area < 7500 and region.area > 400:
                im_location[region.coords[:, 0], region.coords[:, 1]] = True

        if debug:
            plt.subplot(1, 7, 1)
            plt.imshow(im)
            plt.subplot(1, 7, 2)
            plt.imshow(dab_preseg)
            plt.title("dab_preseg")
            plt.subplot(1, 7, 3)
            plt.imshow(aux1)
            plt.title("Before removing holes")
            plt.subplot(1, 7, 4)
            plt.imshow(aux2)
            plt.title("After removing holes")
            plt.subplot(1, 7, 5)
            plt.imshow(mask)
            plt.title("Mask")
            plt.subplot(1, 7, 6)
            plt.imshow(binary_holes)
            plt.title("Before selecting regions")
            plt.subplot(1, 7, 7)
            plt.imshow(im_location)
            plt.title("After selecting regions")
            plt.show()

    return im_location


def gethalo(im, im_labeled, membrane):
    d = 5
    coords = np.argwhere(im_labeled == True)
    r_min = max(np.min(coords[:, 0]) - d, 0)
    r_max = min(np.max(coords[:, 0]) + d, im_labeled.shape[0] - 1)
    c_min = max(np.min(coords[:, 1]) - d, 0)
    c_max = min(np.max(coords[:, 1]) + d, im_labeled.shape[1] - 1)

    sub_im_labeled = im_labeled[r_min:r_max, c_min:c_max]
    sub_im = im[r_min:r_max, c_min:c_max]
    sub_mem = membrane[r_min:r_max, c_min:c_max]

    sub_membrane_complete = np.bitwise_xor(binary_dilation(sub_im_labeled.copy(), disk(d)), sub_im_labeled)
    sub_membrane_complete[sub_mem == 0] = 0

    if 0:
        plt.subplot(151)
        plt.imshow(sub_im)
        plt.subplot(152)
        plt.imshow(sub_im_labeled)
        plt.subplot(153)
        plt.imshow(sub_mem)
        plt.subplot(154)
        plt.imshow(sub_membrane_complete)
        plt.show()

    staining = np.mean(sub_im[sub_membrane_complete > 0].flatten())
    std_staining = np.std(sub_im[sub_membrane_complete > 0].flatten())
    completeness = np.sum(sub_im[sub_membrane_complete > 0].flatten() > staining-std_staining)/ \
                   np.sum(sub_membrane_complete[sub_membrane_complete > 0].flatten())
    return completeness, staining



def membraneHERHotSpot(filename, imgCoords):
    # "No staining"
    json_no_staining = {"applyMatrix": True,
                        "segments": [],
                        "closed": True,
                        "strokeWidth": 1,
                        "fillColor": [255 / 255, 250 / 255, 66 / 255, 0.6],
                        "strokeScaling": False}
    # "Barely percetible and incomplete"
    json_barely_incomplete = {"applyMatrix": True,
                              "segments": [],
                              "closed": True,
                              "strokeWidth": 1,
                              "fillColor": [66 / 255, 183 / 255, 185 / 255, 0.6],
                              "strokeScaling": False}

    # "Weak to moderate and complete"
    json_moderate_complete = {"applyMatrix": True,
                              "segments": [],
                              "closed": True,
                              "strokeWidth": 1,
                              "fillColor": [214 / 255, 145 / 255, 193 / 255, 0.6],
                              "strokeScaling": False}
    # "Intense and complete"
    json_intense_complete = {"applyMatrix": True,
                             "segments": [],
                             "closed": True,
                             "strokeWidth": 1,
                             "fillColor": [199 / 255, 93 / 255, 171 / 255, 0.6],
                             "strokeScaling": False}
    data = []
    scores = [0, 0, 0, 0, '']

    json_dict = [json_no_staining, json_barely_incomplete, json_moderate_complete, json_intense_complete]
    text_dict = ["No staining", "Barely percetible and incomplete",
                 "Weak to moderate and complete", "Intense and complete"]

    slide = openslide.OpenSlide(filename)

    cx = []
    cy = []

    print(imgCoords)
    for cxy in imgCoords:
        cx.append(float(cxy['x']))
        cy.append(float(cxy['y']))

    cx = np.asarray(cx)
    cy = np.asarray(cy)

    offset_x = np.array(int(slide.properties['openslide.bounds-x'])).astype(np.long)
    offset_y = np.array(int(slide.properties['openslide.bounds-y'])).astype(np.long)

    tile_size = (np.max(cx) - np.min(cx)).astype(np.long)
    im = color.rgba2rgb(np.array(slide.read_region((offset_x + np.min(cx).astype(np.long),
                                                    offset_y + np.min(cy).astype(np.long)),
                                                   0, (tile_size, tile_size)), dtype=np.float32) / 255)
    ihc_hed = rgb2hed(im)
    hema = opening(normHema(ihc_hed[:, :, 0]), disk(7))
    hema_original = normHema(ihc_hed[:, :, 0])

    dab = normDab(ihc_hed[:, :, 2])

    mean_hema, std_hema, mean_dab, std_dab = getHemaDabNormParamROI(im, 0.4, 0.1, 0.4, 0.1)


    im_pos = np.zeros(dab.shape)
    max_dab = np.max(dab.flatten())

    # Decidir que algoritmo usar (umbrales con gaussian mixed model)
    gmm_dab = GaussianMixture(n_components=2, covariance_type='full')
    gmm_hema = GaussianMixture(n_components=2, covariance_type='full')
    f_dab = dab.flatten().reshape(-1, 1)
    gmm_dab.fit(f_dab)
    f_hema = hema_original.flatten().reshape(-1, 1)
    gmm_hema.fit(f_hema)

    nc = np.argmax(gmm_dab.weights_)
    th_bg_dab = gmm_dab.means_[nc][0]
    peak_bg_dab = gmm_dab.weights_[nc]
    nc = np.argmin(gmm_dab.weights_)
    th_type_dab = gmm_dab.means_[nc][0]
    peak_type_dab = gmm_dab.weights_[nc]

    nc = np.argmax(gmm_hema.weights_)
    th_bg_hema = gmm_hema.means_[nc][0]
    peak_bg_hema = gmm_hema.weights_[nc]
    nc = np.argmin(gmm_hema.weights_)
    th_type_hema = gmm_hema.means_[nc][0]
    peak_type_hema = gmm_hema.weights_[nc]

    th_3_dab = 0.45
    th_2_dab = 0.38
    th_1_dab = 0.33

    th_3_r = 1.8
    th_2_r = 1.1
    th_1_r = 0.78

    mean_s = []
    mean_h = []

    if debug:
        plt.subplot(1,2,1)
        gmm_x_dab = np.linspace(0, 1.5, 256)
        gmm_y_dab = np.exp(gmm_dab.score_samples(gmm_x_dab.reshape(-1, 1)))
        # Plot histograms and gaussian curves
        plt.hist(f_dab, 255, [0, 1.3], normed=True)
        plt.plot(gmm_x_dab, gmm_y_dab, color="crimson", lw=4, label="GMM")
        plt.subplot(1, 2, 2)
        gmm_x_hema = np.linspace(0, 1.5, 256)
        gmm_y_hema = np.exp(gmm_hema.score_samples(gmm_x_hema.reshape(-1, 1)))
        # Plot histograms and gaussian curves
        plt.hist(f_hema, 255, [0, 1.5], normed=True)
        plt.plot(gmm_x_hema, gmm_y_hema, color="crimson", lw=4, label="GMM")
        plt.show()

    th_dab_ratio = peak_type_dab*abs(th_type_dab-th_bg_dab)/2
    th_hema_ratio = peak_type_hema * abs(th_type_hema-th_bg_hema) / 2

    th_dab_hema_relation = th_dab_ratio/th_hema_ratio

    print(th_bg_dab, th_type_dab, th_type_dab - th_bg_dab, peak_bg_dab, peak_type_dab)
    print(th_bg_hema, th_type_hema, th_type_hema - th_bg_hema, peak_bg_hema,peak_type_hema)
    print(th_dab_hema_relation)
    if th_dab_hema_relation <0.5:  # 0 y 1+    #Pre-segmentacion de nucleos Hema
        hema_preseg = hema_original.copy()
        # Modificar nivel para menor o mayor sensibilidad de nucleos (menor nivel, mayor sensibilidad)
        nivel = 2
        for i in np.arange(0, nivel):
            hema_preseg[hema_preseg <= mean_hema-std_hema] = 0
            mean_hema = np.mean(hema_preseg[hema_preseg > 0].flatten())
            std_hema = np.std(hema_preseg[hema_preseg > 0].flatten())
            if debug:
                plt.subplot(1, 2, 1)
                plt.imshow(im)
                plt.subplot(1, 2, 2)
                plt.imshow(hema_preseg)
                plt.show()

        hema_preseg[hema_preseg > 0] = 1
        hema_preseg = binary_fill_holes(hema_preseg)
        hema_preseg = binary_opening(hema_preseg, disk(7))

        hema_preseg = hema_preseg.astype(dtype=np.bool)
        d = 10
        membrane_complete = np.bitwise_xor(binary_dilation(hema_preseg.copy(), disk(d)), hema_preseg)
        dab_membrane = dab.copy()
        dab_membrane[membrane_complete == 0] = 0

        if debug:
            plt.subplot(1, 2, 1)
            plt.imshow(membrane_complete)
            plt.subplot(1, 2, 2)
            plt.imshow(hema_preseg)
            plt.show()

        nucleus_membrane = label(hema_preseg)
        nucleus_membrane_classified = nucleus_membrane.copy()
        labels_nucleus_membrane_segmentation = np.unique(nucleus_membrane.flatten())
        labels_nucleus_membrane_segmentation = labels_nucleus_membrane_segmentation[
            labels_nucleus_membrane_segmentation > 0]

        for kk, l in enumerate(labels_nucleus_membrane_segmentation):
            aux = np.zeros(hema_preseg.shape, dtype=np.bool)
            aux[nucleus_membrane == l] = True
            c, s = gethalo(dab, aux, membrane_complete)
            h =  np.mean(hema_original[aux].flatten())
            mean_s.append(s)
            mean_h.append(h)
            #print( h, s)

            if s/h > 1:
                if s >0.4:
                    index = 3
                elif s>0.31:
                    index = 2
                else:
                    index =1
            else:
                if s > 0.31:
                    index =1
                else:
                    index = 0

            contours, hierarchy = cv2.findContours(aux.astype(np.uint8) * 255, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)[-2:]
            im_pos[nucleus_membrane == l] = 255
            for i, contour in enumerate(contours):
                scores[index] = scores[index] + 1
                xv = contour[:, 0, 0]
                yv = contour[:, 0, 1]
                data.append({
                    'imgCoords': [
                        {
                            "x": float(xi + np.min(cx)),
                            "y": float(yi + np.min(cy))
                        }
                        for xi, yi in zip(xv, yv)],
                    "path": ["Path", json_dict[index]],
                    "zoom": "0.016597",
                    "name": text_dict[index]})
    else:  # 2+ y 3+
        dab = opening(normDab(ihc_hed[:, :, 2]), disk(7))
        dab_original = normDab(ihc_hed[:, :, 2])

        # Segmentacion general de membrana
        th_dab = th_type_dab / 2 + th_bg_dab /2

        membrane = dab_original > th_dab
        membrane_nso = remove_small_objects(membrane, min_size=5000)
        membrane_dil = binary_closing(membrane_nso, disk(15))
        membrane_nh = binary_fill_holes(membrane_dil)
        membrane_nh2 = binary_closing(membrane_nh.copy(), disk(7))

        aux_dab = dab_original.copy() - hema_original.copy()
        aux_dab = normImage(aux_dab)
        aux_dab[membrane_nh2 == 0] = 0

        # Segmentacion de huecos en membrana
        nucleus_membrane = membraneDabBioMarkerROI(im, aux_dab, membrane_nh2)
        nucleus_membrane = label(nucleus_membrane)
        labels_nucleus_membrane_segmentation = np.unique(nucleus_membrane.flatten())
        labels_nucleus_membrane_segmentation = labels_nucleus_membrane_segmentation[
            labels_nucleus_membrane_segmentation > 0]

        for kk, l in enumerate(labels_nucleus_membrane_segmentation):
            aux = np.zeros(nucleus_membrane.shape, dtype=np.bool)
            aux[nucleus_membrane == l] = True
            c, s = gethalo(dab, aux, membrane)
            h= np.mean(hema[aux].flatten())
            #print( h,s)
            mean_s.append(s)
            mean_h.append(h)

            if s / h > 1:
                if s > 0.4:
                    index = 3
                elif s > 0.31:
                    index = 2
                else:
                    index = 1
            else:
                if s > 0.31:
                    index = 1
                else:
                    index = 0

            contours, hierarchy = cv2.findContours(aux.astype(np.uint8) * 255, cv2.RETR_TREE,
                                                   cv2.CHAIN_APPROX_SIMPLE)[-2:]
            im_pos[nucleus_membrane == l] = 255
            for i, contour in enumerate(contours):
                scores[index] = scores[index] + 1
                xv = contour[:, 0, 0]
                yv = contour[:, 0, 1]
                data.append({
                    'imgCoords': [
                        {
                            "x": float(xi + np.min(cx)),
                            "y": float(yi + np.min(cy))
                        }
                        for xi, yi in zip(xv, yv)],
                    "path": ["Path", json_dict[index]],
                    "zoom": "0.016597",
                    "name": text_dict[index]})

        if debug:
            plt.subplot(251)
            plt.imshow(im)
            plt.subplot(252)
            plt.imshow(dab_original)
            plt.subplot(253)
            plt.imshow(dab)
            plt.subplot(254)
            plt.imshow(membrane)
            plt.subplot(255)
            plt.imshow(nucleus_membrane)
            plt.subplot(256)
            plt.imshow(membrane_nh2)
            plt.subplot(257)
            plt.imshow(im_pos)
            plt.show()

    print(np.mean(np.array(mean_h)))
    print(np.mean(np.array(mean_s)))
    print(np.mean(np.array(mean_s))/np.mean(np.array(mean_h)))
    total = np.sum(np.array(scores[0:3]))
    if total ==0:
        total = 1
    if scores[3] / total > 0.1:
        scores[4] = '3+'
    elif scores[2] / total > 0.1:
        scores[4] = '2+'
    elif scores[1] / total > 0.1:
        scores[4] = '1+'
    else:
        scores[4] = '0'

    return data, scores


if __name__ == '__main__':
    # filename = '/media/telemed/BDPATH_WEBIM1/PI0032WEB/P002-HE-033-2.mrxs'
    # nuclearBioMarker(filename)
    # imgCoords = [{'x': 61817.16796074552, 'y': 11858.25930202452}, {'x': 63817.16796074552, 'y': 11858.25930202452}, {'x': 63817.16796074552, 'y': 13858.25930202452}, {'x': 61817.16796074552, 'y': 13858.25930202452}]
    # ki67HotSpot(filename, imgCoords)
    '''
    filename = 'E:/BDPATH_WEBIM1/PI0032WEB/P002-ECAD-033-2.mrxs'
    imgCoords = [{'x': 34308.02293987662, 'y': 75884.7299117593}, {'x': 35308.02293987662, 'y': 75884.7299117593}, {'x': 35308.02293987662, 'y': 76884.7299117593}, {'x': 34308.02293987662, 'y': 76884.7299117593}]
    filename = 'E:/BDPATH_WEBIM1/PI0032WEB/P004-ECAD-042-1.mrxs'
    imgCoords = [{'x': 47595.597922910216, 'y': 105386.43386603032}, {'x': 48595.597922910216, 'y': 105386.43386603032}, {'x': 48595.597922910216, 'y': 106386.43386603032}, {'x': 47595.597922910216, 'y': 106386.43386603032}]
    membraneBioMarkerHotSpot(filename, imgCoords)

    '''

    print("0")
    imgCoords = [{'x': 57752.62666027365, 'y': 54068.82266015419}, {'x': 58752.62666027365, 'y': 54068.82266015419}, {'x': 58752.62666027365, 'y': 55068.82266015419}, {'x': 57752.62666027365, 'y': 55068.82266015419}]
    filename = '/media/telemed/BDPATH_WEBIM1/PI0032WEB/P029-HER-097-3.mrxs'
    membraneHERHotSpot(filename, imgCoords)

    print("0")
    imgCoords = [{'x': 42397.41969083141, 'y': 40339.46495259131}, {'x': 43397.41969083141, 'y': 40339.46495259131}, {'x': 43397.41969083141, 'y': 41339.46495259131}, {'x': 42397.41969083141, 'y': 41339.46495259131}]
    
    filename = '/media/telemed/BDPATH_WEBIM1/PI0032WEB/P071-HER-235-V.mrxs'
    membraneHERHotSpot(filename, imgCoords)

    print("1+")
    imgCoords = [{'x': 19873.777468542103, 'y': 80851.76595907542}, {'x': 20873.777468542103, 'y': 80851.76595907542},
                 {'x': 20873.777468542103, 'y': 81851.76595907542}, {'x': 19873.777468542103, 'y': 81851.76595907542}]
    filename = '/media/telemed/BDPATH_WEBIM1/PI0032WEB/P017-HER-069-2.mrxs'
    membraneHERHotSpot(filename, imgCoords)
    
    print("2+")
    imgCoords = [{'x': 44193.665219742616, 'y': 49884.791757154875}, {'x': 45193.665219742616, 'y': 49884.791757154875}, {'x': 45193.665219742616, 'y': 50884.791757154875}, {'x': 44193.665219742616, 'y': 50884.791757154875}]
    filename = '/media/telemed/BDPATH_WEBIM1/PI0032WEB/P202-HER-351-3.mrxs'
    membraneHERHotSpot(filename, imgCoords)
    
    print("3+")
    imgCoords = [{'x': 35928.98208672112, 'y': 83539.35500985188}, {'x': 36928.98208672112, 'y': 83539.35500985188}, {'x': 36928.98208672112, 'y': 84539.35500985188}, {'x': 35928.98208672112, 'y': 84539.35500985188}]
    filename =  '/media/telemed/BDPATH_WEBIM1/PI0032WEB/P005-HER-047-6.mrxs'
    membraneHERHotSpot(filename, imgCoords)
    '''


    imgCoords = [{'x': 45805.327906384024, 'y': 26007.084279484992}, {'x': 46805.327906384024, 'y': 26007.084279484992}, {'x': 46805.327906384024, 'y': 27007.084279484992}, {'x': 45805.327906384024, 'y': 27007.084279484992}]
    filename = '/media/telemed/BDPATH_WEBIM1/PI0032WEB/P176-HER-087-1.mrxs'
    membraneHERHotSpot(filename, imgCoords)
    '''





