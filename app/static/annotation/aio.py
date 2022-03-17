#import xmltodict
import numpy as np

from imageio import imwrite
import cv2
import os
from openslide import OpenSlide
import json



dictionary_types = {
    'ROI_TUMORAL': 1,
    'ROI_tumoral': 1,
    'ROI_Tumoral': 1,
    'ROI': 1,
    'ROI_NONTUMORAL': 2,
    'ROI_NonTumoral': 2,
    'ROI_nontumoral': 2,
    'DCIS': 3,
    'IDC': 4,
    'LCIS': 5,
    'ILC': 6,
    'OtherTumor': 7,
    'OTHERTUMOR': 7,
    'Epithelium': 8,
    'EPITHELIUM': 8,
    'Artifacts': 9,
    'ARTIFACTS': 9,
    'Fat': 10,
    'FAT': 10,
    'Vessels': 11,
    'VESSELS': 11}


def findExtension(directory, extension='.xml'):
    files = []    
    for file in os.listdir(directory):
        if file.endswith(extension):
            files += [file]
    files.sort()
    return files

def fillImage(image, coordinates,color=255):
   cv2.fillPoly(image, coordinates, color=color)
   return image


def readXML(filename):

    with open(filename) as fd:
        doc = xmltodict.parse(fd.read())
    pixel_spacing =doc['Annotations']['@MicronsPerPixel']
    annotations = doc['Annotations']['Annotation']
    labels = []
    zooms = []

    coords = []
    texts = []
    for i, annotation in enumerate(annotations):
        try:
            regions = annotation['Regions']['Region']
            for j, region in enumerate(regions):
                vertices = region['Vertices']['Vertex']
                coord = []
                for vertex in vertices:
                    x = float(vertex['@X'])
                    y = float(vertex['@Y'])
                    coord += [[x, y]]
                coords += [coord]
                texts += [annotations[i]['@Name']]
                zooms += [region['@Zoom']]
                t = (annotations[i]['@Name']).upper()
                labels += [dictionary_types[str(t)]]
        except:
            pass
    labels = np.asarray(labels)
    coords = np.asarray(coords)

    indices_roi = [index for index, label in enumerate(labels) if label == 1]
    indices_tumor = [index for index, label in enumerate(labels) if label > 2]

    for index_roi in indices_roi:
        labels[index_roi] = 1
        texts[index_roi] = 'ROI_NonTumoral'
        for index_tumor in indices_tumor:
            if evaluateIntersection(coords[index_roi], coords[index_tumor]):
                labels[index_roi] = dictionary_types['ROI_Tumoral']
                texts[index_roi] = 'ROI_Tumoral'
                break
    return coords, labels, texts, zooms

def evaluateIntersection(roi, tumor):
    roi_coords = np.array([np.array(xi) for xi in roi])
    tumor_coords = np.array([np.array(xi) for xi in tumor])

    min_x = np.min(roi_coords[:, 0])
    max_x = np.max(roi_coords[:, 0])
    min_y = np.min(roi_coords[:, 1])
    max_y = np.max(roi_coords[:, 1])
    inlier = 0
    for i in np.arange(0, tumor_coords.size/2):
        if tumor_coords[int(i), 0] >= min_x and tumor_coords[int(i), 0] <= max_x:
            if tumor_coords[int(i), 1] >= min_y and tumor_coords[int(i), 1] <= max_y:
                inlier = 1;
                break;
    return inlier

def readJSON(filename, _dictionary_types=dictionary_types):
    json_data = open(filename).read()
    data = json.loads(json_data)
    labels = []
    coords = []
    texts = []

    for i, annotation in enumerate(data):
        texts += [data[i]['name']]
        labels += [_dictionary_types[data[i]['name'].upper()]]
        coord = []
        for j, cxy in enumerate(data[i]['imgCoords']):
            x = float(cxy['x'])
            y = float(cxy['y'])
            coord += [[x, y]]
        coords += [coord]
    labels = np.asarray(labels)
    coords = np.asarray(coords)
    return coords, labels, texts

def saveImage(filename, image_size, coordinates, labels, sample=8):

    unique_labels = np.unique([dictionary_types[f] for f in dictionary_types])
    img = np.zeros((image_size[0], image_size[1]), dtype=np.uint8)
    for c, l in zip(coordinates, labels):
        if l == 1 or l > 2:
            img1 = fillImage(img, [np.int32(np.stack(c))], color=int(l / len(unique_labels) * 255))
            img2 = img1[::sample, ::sample]
    imwrite(filename, img2)

if __name__=='__main__':
    

        path = 'C:/Users/BlancaPT/PycharmProjects/Reading_WSI/PI0032/app/static/wsi/397W_HE_40x.svs'
        scan = OpenSlide(path)
        dims = scan.dimensions
        img_size = (dims[1], dims[0], 3)
        path_xml = 'C:/Users/BlancaPT/PycharmProjects/Reading_WSI/PI0032/app/static/wsi/397W_HE_40x.xml'

        coords, labels, texts = readXML(path_xml)

        saveImage('C:/Users/BlancaPT/PycharmProjects/Reading_WSI/PI0032/app/static/wsi/397W_HE_40x.png',
                  img_size, coords, labels)



