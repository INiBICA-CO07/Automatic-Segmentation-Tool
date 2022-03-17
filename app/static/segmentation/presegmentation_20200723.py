import numpy as np
import openslide

from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import cv2
from skimage.segmentation import watershed
from skimage import color
from skimage.filters import sobel
from skimage.color import rgb2hed
from skimage.morphology import disk, square
from skimage.filters.rank import gradient
from skimage.morphology import remove_small_objects, remove_small_holes,  closing, opening, erosion, dilation, binary_dilation
from skimage.measure import label, regionprops, find_contours
from skimage.filters import gaussian
from skimage.filters import rank
from skimage.transform import rescale
from skimage.color import label2rgb


def extractMainTissue(im):
    maskIm = np.invert(np.logical_and(np.logical_and(im[:, :, 0] == 0, im[:, :, 1] == 0),
                                      im[:, :, 2] == 0))
    labelIm = label(maskIm)
    rp = regionprops(labelIm)
    centroids = [region.local_centroid for region in rp] #Tuple: (row, col)
    distances = []
    for d in centroids:
        distances = (d[0] - im.shape[0])**2 + (d[1] - im.shape[1])**2

    i_min_d = np.argmin(distances)
    imgBG = np.zeros(maskIm.shape, dtype=np.bool)
    imgBG[rp[i_min_d].coords[:, 0], rp[i_min_d].coords[:, 1]] = True
    lowResFG = np.copy(im[:, :, 0:3]) * imgBG[:, :, np.newaxis].astype(dtype=np.float)

    contours, hierarchy = cv2.findContours(imgBG.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return lowResFG, contours


def getScaleFactor(scan, level_inicial, level_final):
    dimi = scan.level_dimensions[level_inicial]
    dimf = scan.level_dimensions[level_final]
    factor_x = np.round(float(dimf[0]) / float(dimi[0]))
    factor_y = float(dimf[1]) / float(dimi[1])
    return [factor_x, factor_y]


def normImage(im, max_=[], min_=[]):
    if max_:
        im = (im - min_) / (max_ - min_)
    else:
        im = (im - np.min(im)) / (np.max(im) - np.min(im))
    return im




def postcontours (c):
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
            p1 = [xv[i-1], yv[i-1]]
        if i >= xv.size-2:
            if i == xv.size-2:
                p3 = [xv[i + 1], yv[i + 1]]
            elif i == xv.size-1:
                p3 = [xv[0], yv[0]]
        else:
            p3 = [xv[i + 1], yv[i + 1]]

        #Linea post
        dxl = p3[0] - p2[0]
        dyl = p3[1] - p2[1]
        #Linea prev
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
    newc = np.vstack((newc, newc[0 , :]))
    return newc

def ki67ROI(im, th):
    im_binary = im > th

    im_aux = im.copy()
    im_aux[im_binary == 0] = 0

    elevation_map = sobel(im_aux)
    segmentation = watershed(elevation_map)



    im_holes = remove_small_holes(im_binary, area_threshold=2000, connectivity=1, in_place=False)
    im_smallremoved = remove_small_objects(im_holes, min_size=100)
    im_label = label(im_smallremoved)

    regions = regionprops(im_label)
    area = [region.area for region in regions]
    area_m = np.mean(area)

    im_label_filtered = im_label.copy()
    '''
    for region in regions:
      area = region.area
      ecc = region.eccentricity
      #if area < area_m and ecc > 0.97:
      if ecc > 0.97:
          im_label_filtered[region.coords[:,0], region.coords[:,1]] = 0
    '''
    return im_label_filtered, segmentation

def ki67HotSpot(filename, imgCoords):
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
                                         0, (tile_size, tile_size)), dtype=np.float32)/255)

    data = []
    ihc_hed = rgb2hed(im)
    ihc_hed = (ihc_hed + 1) / 2
    hema = opening(ihc_hed[:, :, 0], disk(7))
    dab = opening(ihc_hed[:, :, 2], disk(7))

    hema_nucleus, segmentation = ki67ROI(hema, 0.25)

    dab_nucleus, segmentation = ki67ROI(dab, 0.3)

    '''
    plt.subplot(131)
    plt.imshow(hema)
    plt.subplot(132)
    plt.imshow(hema_nucleus)
    plt.subplot(133)
    plt.imshow(segmentation)
    plt.show()
    '''

    contours, hierarchy = cv2.findContours(hema_nucleus.astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

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
    contours, hierarchy = cv2.findContours(dab_nucleus.astype(np.uint8), cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)
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
            "name": "Positive Nuclei"})
    p_c = len(contours)
    if p_c + n_c == 0:
        score = 0
    else:
        score = p_c / (p_c + n_c)
    return data, score

def ki67(filename):

    #Funcion colormap general

    slide = openslide.OpenSlide(filename)
    level_lowRes = slide.level_count - 1
    level_highRes = level_lowRes - 8

    lowRes = np.asarray(slide.read_region((0, 0), level_lowRes,
                                          slide.level_dimensions[level_lowRes]),
                        dtype=np.float32) / 255

    mainTissue, contours = extractMainTissue(lowRes)

    print("Level low res", level_lowRes)
    print("Level high res", level_highRes)

    offset_x = int(slide.properties['openslide.bounds-x'])
    offset_y = int(slide.properties['openslide.bounds-y'])

    cxmin = np.min(contours[0][:, 0, 0])
    cymin = np.min(contours[0][:, 0, 1])
    cxmax = np.max(contours[0][:, 0, 0])
    cymax = np.max(contours[0][:, 0, 1])

    clim = np.asarray([cxmin, cymin, cxmax, cymax]).astype(np.long)

    factor_lowtohighres = getScaleFactor(slide, level_lowRes, level_highRes)

    max_dim = 500
    wx = (cxmax - cxmin) * factor_lowtohighres[0]
    wy = (cymax - cymin) * factor_lowtohighres[1]

    x_vector = np.arange(0, np.ceil(wx / max_dim) + 1) * max_dim
    y_vector = np.arange(0, np.ceil(wy / max_dim) + 1) * max_dim

    x_vector[x_vector > wx] = wx
    y_vector[y_vector > wy] = wy

    factor_hightomaxres = getScaleFactor(slide, level_highRes, 0)

    x_vector = x_vector.astype(dtype=np.int32)
    y_vector = y_vector.astype(dtype=np.int32)

    coordenadas_x = []
    coordenadas_y = []
    opacity = []

    wx = int(x_vector[1] - x_vector[0])
    wy = int(y_vector[1] - y_vector[0])

    im_negative = np.zeros((y_vector.size * 2, x_vector.size * 2))
    im_positive = np.zeros((y_vector.size * 2, x_vector.size * 2))

    for py, y in enumerate(y_vector[:-1]):
        print(py, '/', len(y_vector[:-1]))
        for px, x in enumerate(x_vector[:-1]):
                cx = int(offset_x + (clim[0] + x) * factor_hightomaxres[0])
                cy = int(offset_y + (clim[1] + y) * factor_hightomaxres[1])
                im = color.rgba2rgb(np.asarray(slide.read_region((cx, cy), level_highRes, (wx, wy)),
                                               dtype=np.float32) / 255)

                if np.sum(im[:, :, 0].flatten() > 0.9) < im.shape[0] * im.shape[1] * 0.95:
                    ihc_hed = rgb2hed(im)
                    ihc_hed = (ihc_hed + 1) / 2
                    dab = ihc_hed[:, :, 2]
                    hema = ihc_hed[:, :, 0]

                    selection_element = disk(3)  # matrix of n pixels with a disk shape

                    dab_sharpness = gradient(dab, selection_element)
                    dab_blur = gaussian(dab, sigma=1.5)

                    hema_sharpness = gradient(hema, selection_element)
                    hema_blur = gaussian(hema, sigma=1.5)

                    im_positive_aux = np.bitwise_and(dab_blur > 0.3, dab_sharpness >= 13)
                    im_negative_aux = np.bitwise_and(hema_blur > 0.25, hema_sharpness >= 12)

                    sum_positive = np.sum(im_positive_aux.flatten())
                    sum_negative = np.sum(im_negative_aux.flatten())


                    if sum_negative > sum_positive:
                        im_positive[py * 2:py * 2 + 2, px * 2:px * 2 + 2] = sum_positive
                    else:
                        im_positive[py * 2:py * 2 + 2, px * 2:px * 2 + 2] = 0

                    im_negative[py * 2:py * 2 + 2, px * 2:px * 2 + 2] = sum_negative


    selem = disk(2)
    q = np.linspace(0, 1, 10)

    p80 = np.percentile(im_negative[im_negative > 0].flatten(), 80)

    indices_no_positive = im_positive < q[1]
    indices_no_negative_enough = im_negative < p80

    divisor = im_positive + im_negative
    divisor[divisor == 0] = 1

    im_ratio = im_positive/divisor

    im_ratio = normImage(im_ratio)
    im_ratio[indices_no_positive] = 0
    im_ratio[indices_no_negative_enough] = 0
    im_ratio = rank.mean(im_ratio, selem=selem)


    im_ratio = rescale(im_ratio, 0.5)
    im_ratio = rescale(im_ratio, 2, order=0)



    contours = []

    for i, q_ in enumerate(q[:-1]):
        im_ratio[np.logical_and(im_ratio > q[i], im_ratio <= q[i+1])] = q[i+1]
        im_binary = np.zeros(im_ratio.shape)
        im_binary[np.logical_and(im_ratio > q[i], im_ratio <= q[i+1])] = 1

        out = label(im_binary, connectivity=1)
        contour = find_contours(im_binary, level=0.5,  fully_connected='low', positive_orientation="high")

        for kk, c in enumerate(contour):
            c = postcontours(c)
            contours.append(c)
            xv = ((clim[0] + c[:, 1]/2*wx) * factor_hightomaxres[0]).astype(dtype=np.uint)
            yv = ((clim[1] + c[:, 0]/2*wy) * factor_hightomaxres[1]).astype(dtype=np.uint)
            coordenadas_x.append(xv)
            coordenadas_y.append(yv)
            opacity.append(q[i+1])

    opacity_array = np.asarray(opacity)

    min_ = np.min(opacity_array)
    max_ = np.max(opacity_array)
    data = []

    for (xv, yv, opa) in zip(coordenadas_x, coordenadas_y, opacity):
        aux = {"applyMatrix": True,
               "segments": [],
               "closed": True,
               "fillColor": [245/255., 206/255., 66/255., (opa - min_) / (max_ - min_) * 0.8],
               "strokeWidth": 0,
               "strokeColor": [0, 0, 0],
               "strokeAlpha": 0,
               "strokeScaling": False}
        data.append({
            'imgCoords': [
                {
                    "x": float(xi),
                    "y": float(yi)
                }
                for xi, yi in zip(xv, yv)],
            "path": ["Path", aux],
            "zoom": "0.016597",
            "name": "KI67 Colormap"})

    return data

if __name__ == '__main__':
    filename = '/media/telemed/BDPATH_WEBIM1/PI0032WEB/P025-Ki67-337-9.mrxs'
    imgCoords = [{'x': 61817.16796074552, 'y': 11858.25930202452}, {'x': 63817.16796074552, 'y': 11858.25930202452}, {'x': 63817.16796074552, 'y': 13858.25930202452}, {'x': 61817.16796074552, 'y': 13858.25930202452}]
    ki67HotSpot(filename, imgCoords)

