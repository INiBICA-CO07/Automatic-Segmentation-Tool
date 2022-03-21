from flask import abort, make_response, render_template, url_for, request, jsonify, g, send_from_directory
import requests
from flask_appbuilder import BaseView, expose, has_access, ModelView, MasterDetailView
from flask_appbuilder.models.sqla.interface import SQLAInterface
from io import BytesIO
import numpy as np
from skimage import color
import cv2

from openslide import OpenSlideError
import openslide

import os, logging
import os.path
import json
from datetime import datetime
import app.static.annotation.aio as aio
import app
from .models import Patient, Wsifile, WsifileTest, Annotationfile, Staining, Variant, Cnv
from app import db, appbuilder
import app.static.deeplab.segmentROI as segmentROI
import app.static.segmentation.presegmentation as presegmentation



class PILBytesIO(BytesIO):
    def fileno(self):
        '''Classic PIL doesn't understand io.UnsupportedOperation.'''
        raise AttributeError('Not supported')


def _get_slide(path):


    path = path.replace("\\", "/")

    path = os.path.abspath(os.path.join(app.app.basedir, path))
    if not path.startswith(app.app.basedir + os.path.sep):
        # Directory traversal
        abort(404)
    if not os.path.exists(path):
        abort(404)
    try:
        slide = app.app.cache.get(path)
        slide.filename = os.path.basename(path)
        return slide
    except OpenSlideError:
        abort(404)


class Annotation(BaseView):

    @expose('/<path:path>')
    @has_access
    def slide(self, path):
        slide = _get_slide(path)
        slide_url = url_for('dzi', path=path)
        return render_template('slide-fullpage.html', slide_url=slide_url,
                               slide_filename=slide.filename, slide_mpp=slide.mpp)

    @expose('/<path:file_path>/<path:annotation_path>')
    @has_access
    def slide(self, file_path, annotation_path):
        datamodel = SQLAInterface(Wsifile, Annotationfile)
        queryWSI = db.session.query(Wsifile).filter_by(url_path=file_path)
        if queryWSI.first() is None:
            queryWSI = db.session.query(WsifileTest).filter_by(url_path=file_path)
        staining = queryWSI.first().staining_id
        if staining == 6:
            dictionary = "ki67.json"
        elif staining == 2:
            dictionary = "ER.json"
        elif staining == 3:
            dictionary = "PR.json"
        elif staining == 5:
            dictionary = "p53.json"
        elif staining ==7:
            dictionary = "ECAD.json"
        elif staining ==9:
            dictionary = "HER2.json"
        else:
            dictionary = "default.json"

        slide = _get_slide(file_path)
        slide_url = url_for('Annotation.dzi', path=file_path)

        self.update_redirect()

        return self.render_template('as_viewer.html', file_name=file_path, annotation_path=annotation_path,
                                    slide_url=slide_url, slide_mpp=slide.mpp,
                                    dictionary=dictionary)

    @expose('/<path:path>.dzi')
    @has_access
    def dzi(self, path):
        slide = _get_slide(path)
        format = app.app.config['DEEPZOOM_FORMAT']
        resp = make_response(slide.get_dzi(format))
        resp.mimetype = 'application/xml'
        return resp

    @expose('/<path:path>_files/<int:level>/<int:col>_<int:row>.<format>')
    def tile(self, path, level, col, row, format):
        slide = _get_slide(path)
        format = format.lower()
        if format != 'jpeg' and format != 'png':
            # Not supported by Deep Zoom
            abort(404)
        try:
            tile = slide.get_tile(level, (col, row))
        except ValueError:
            # Invalid level or coordinates
            abort(404)

        buf = PILBytesIO()
        tile.save(buf, format, quality=app.app.config['DEEPZOOM_TILE_QUALITY'])
        resp = make_response(buf.getvalue())
        resp.mimetype = 'image/%s' % format
        return resp


    @expose('/getWsiInfo', methods=['POST'])
    @has_access
    def getWsiInfo(self):
        datamodel = SQLAInterface(Wsifile, Annotationfile)
        dict = request.form
        slide = dict.get('slide', default='')
        queryWSI = db.session.query(Wsifile).filter_by(url_path=slide)
        wsifile = queryWSI.first()
        return jsonify(cellularity=wsifile.tumor_cellularity, comments=wsifile.comments, staining_id=wsifile.staining_id)

    @expose('/saveWsiInfo', methods=['POST'])
    @has_access
    def saveWsiInfo(self):
        datamodel = SQLAInterface(Wsifile, Annotationfile)
        dict = request.form
        slide = dict.get('slide', default='')
        cellularity = dict.get('cellularity', default='')
        comments = dict.get('comments', default='')
        queryWSI = db.session.query(Wsifile).filter_by(url_path=slide)
        wsifile = queryWSI.first()

        wsifile.comments = comments
        wsifile.tumor_cellularity = cellularity
        try:
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            logging.warning(e)
            return 'Error'
        return 'Ok'

    @expose('/saveJson', methods=['POST'])
    @has_access
    def saveJson(self):
        datamodel = SQLAInterface(Wsifile, Annotationfile)
        dict = request.form
        slide = dict.get('slide', default='')
        annotationName = dict.get('annotation_name', default='') + '.json'
        annotationType = dict.get('annotation_type', default='')
        validation = dict.get('validation', default='')
        if validation == "true":
            validation = True
        else:
            validation = False
        queryWSI = db.session.query(Wsifile).filter_by(url_path=slide)
        wsifile = queryWSI.first()

        # Comprobar si existe la anotacion
        queryAnn = db.session.query(Annotationfile).filter_by(url_path=annotationName)

        if queryAnn.first() == None:

            db.session.add(Annotationfile(url_path=annotationName,
                                          created_by=g.user.username,
                                          wsifile=wsifile,
                                          modified_on=datetime.now(),
                                          modified_by=g.user.username,
                                          validated=validation,
                                          annotationtype_id=int(annotationType)))
        else:
            queryAnn.first().modified_on = datetime.now()
            queryAnn.first().modified_by = g.user.username
            queryAnn.first().annotationtype_id = int(annotationType)
            queryAnn.first().validated = validation
        try:
            db.session.commit()
            json = dict.get('json', default='{}').encode('utf-8')
            if len(annotationName) > 0:
                with open(os.path.join(app.app.annotationdir, annotationName), 'wb+') as file:
                    file.write(json)
            return annotationName
        except Exception as e:
            db.session.rollback()
            logging.warning(e)
            return 'Error'

    @expose('/loadJson')
    @has_access
    def loadJson(self):
        source = os.path.join(app.app.annotationdir, request.args.get('src', ''))
        if os.path.isfile(source):
            with open(source, 'r') as file:
                content = file.read()
                return jsonify(content)
        else:
            return jsonify('[]')

    @expose('/createDictionary')
    @has_access
    def createDictionary(self):
        name = request.args.get('name', '')
        slide = request.args.get('slide')
        path = os.path.join(app.app.appdir, 'app/static/dictionaries', name)
        if os.path.isfile(path):
            # dictionary already exists
            return 'error'
        else:
            with open(path, 'w+') as dictionary:
                dictionary.write("[]")

        with open(os.path.join(app.app.slidedictionaries), 'r') as file:
            dictioary_map = json.loads(file.read())
        dictioary_map[slide] = name
        with open(os.path.join(app.app.slidedictionaries), 'w') as file:
            file.write(json.dumps(dictioary_map))

        respone = '{"name":"' + name + '", "path":"/' + path + '"}';
        return respone

    @expose('/getDictionaries')
    @has_access
    def getDictionaries(self):
        dir = os.path.join(app.app.appdir, 'app/static/dictionaries')
        if os.path.isfile(dir):
            # no dictionaries found
            return '-1'
        else:
            return json.dumps(os.listdir(dir))

    @expose('/static/dictionaries/<dictionary>')
    @has_access
    def loadDictionary(self, dictionary):
        dictionary = os.path.join(app.app.appdir, 'app/static/dictionaries', dictionary)
        if os.path.isfile(dictionary):
            # no dictionary found
            return '404'
        else:
            # return dictionary
            with open(dictionary, 'r') as file:
                return json.dumps(file.read())

    @expose('/switchDictionary')
    @has_access
    def switchDictionary(self):
        name = request.args.get('name', '')
        slide = request.args.get('slide')

        with open(os.path.join(app.app.slidedictionaries), 'r') as file:
            dictioary_map = json.loads(file.read())
        dictioary_map[slide] = name
        with open(os.path.join(app.app.slidedictionaries), 'w') as file:
            file.write(json.dumps(dictioary_map))

        return '200'

    @expose("/importAnnotationXML")
    @has_access
    def importAnnotationXML(self):
        source = os.path.join(app.app.annotationdir, request.args.get('src', ''))
        if os.path.isfile(source):
            try:

                coords, labels, texts, zooms = aio.readXML(source)
                return jsonify(coords=coords.tolist(), labels=labels.tolist(), texts=texts, zooms=zooms)
            except ImportError:
                print("ERROR: annotation script  not found!")
                return "404"
        else:
            return '[]'

    @expose('/checkAnnotationExistence')
    @has_access
    def checkAnnotationExistence(self):
        datamodel = SQLAInterface(Annotationfile)
        name = request.args.get('src', '')
        queryAnn = db.session.query(Annotationfile).filter_by(url_path=name + '.json')
        if queryAnn.first() == None:
            out = "False"
        else:
            out = ""
            if queryAnn.first().annotationtype_id == 1:
                out = "Detailed ROI segmentation for algorithm training"
            elif queryAnn.first().annotationtype_id == 2:
                out = "Marking of high tumor purity for NGS sequencing"

        return out

    @expose('/getUsername')
    @has_access
    def getUsername(self):
        return g.user.username

    @expose("/getAnnotationFiles")
    @has_access
    def getAnnotationFiles(self):
        datamodel = SQLAInterface(Wsifile)
        root_name = request.args.get('src', '')
        query = db.session.query(Wsifile).filter_by(url_path=root_name)
        annotation_files = query.first().annotationfile
        files = [str(f.url_path) for f in annotation_files]
        return jsonify(files)

    @expose("/autoprocessing", methods=['POST'])
    @has_access
    def autoprocessing(self):
        datamodel = SQLAInterface(Wsifile, Staining)
        dict = request.form

        slide = os.path.abspath(os.path.join(app.app.basedir, dict.get('slide', default='')))
        slide_for_dgx = dict.get('slide', default='')
        staining = db.session.query(Wsifile).filter_by(url_path=dict.get('slide', default='')).first().staining_id
        staining_name = db.session.query(Staining).filter_by(id=staining).first().name

        print(staining)
        if staining == 1: #HE
            imgCoords = json.loads(dict.get('imgCoords', default=''))
            model = segmentROI.DeepLabModel(app.app.deeplabmodelfile)
            # model = segmentROICRF.DeepLabModel(app.app.deeplabmodelfilecrf)
            out = json.dumps(model.runROI(slide, imgCoords))
        elif staining == 6 or staining == 3 or staining == 5 or staining == 2: #6: ki67; 3: PR; 2: ER; 5: p53
            processingType = dict.get('processingType', default='')
            if (processingType == 'global'): #colormap
                out = json.dumps(presegmentation.nuclearBioMarker(slide, staining_name))
            else: #ROI
                imgCoords = json.loads(dict.get('imgCoords', default=''))
                print(imgCoords)
                test_url = app.app.processinghost + '/nuclearhotspot'
                print(test_url)
                response = requests.post(test_url, json={"image": slide_for_dgx, "imgCoords": imgCoords})
                # decode response
                out = response.text


        elif staining == 7 or staining == 9:  # 7: ECAD
            processingType = dict.get('processingType', default='')
            if (processingType == 'global'):  # colormap
                out = json.dumps([])
            else:  # ROI
                imgCoords = json.loads(dict.get('imgCoords', default=''))
                test_url = app.app.processinghost + '/membranehotspot'
                response = requests.post(test_url, json={"image": slide_for_dgx, "imgCoords": imgCoords})
                # decode response
                out = response.text

        return out


class AnnotationFilesModelView(ModelView):
    show_title = "Annotation File"
    list_title = "Annotation File List"
    edit_exclude_columns = ['url_path', 'wsifile', 'created_by', 'modified_on', 'modified_by', 'annotationtype']
    datamodel = SQLAInterface(Annotationfile)
    label_columns = {'open_annotation': 'Filename'}
    list_columns = ['open_annotation', 'created_by', 'modified_by', 'modified_on', 'validated', 'annotationtype']


class AnnotationFilesGeneralModelView(ModelView):
    show_title = "Annotation File"
    list_title = "Annotation File List"
    edit_exclude_columns = ['url_path', 'wsifile', 'created_by', 'modified_on', 'modified_by']
    datamodel = SQLAInterface(Annotationfile)
    label_columns = {'open_annotation': 'Filename', 'wsifile.staining': 'Staining', 'wsifile.type': 'Type',
                     'annotationtype': 'Annotation type'}
    list_columns = ['open_annotation', 'wsifile', 'wsifile.staining', 'wsifile.type', 'created_by', 'modified_on',
                    'validated', 'annotationtype']


class ProjectFilesModelView(ModelView):
    datamodel = SQLAInterface(Wsifile)
    list_title = "WSI File List"
    show_title = "WSI File"
    base_permissions = ['can_list', 'can_show']
    label_columns = {'annotate': 'WSI Name', 'added_on': 'Added on', 'annotationfile': 'Annotation Files',
                     'comments': 'Comments'}
    list_columns = ['annotate', 'staining', 'type', 'added_on', 'annotationfile', 'comments']
    related_views = [AnnotationFilesModelView]
    show_template = 'appbuilder/general/model/show_cascade.html'

class VariantsModelView(ModelView):
    datamodel = SQLAInterface(Variant)
    list_title = "Detected Variant List"
    show_title = "Detected Variant List"
    base_permissions = ['can_list', 'can_show']
    label_columns = {'gene': 'Genes', 'category': 'Variant Class', 'freq_allelic': 'Var Freq. (%)',
                      'variant_effect': 'Variant effect', 'Clinvar': 'Clin Var', 'aminoacid_change':'Aminoacid Change'}
    list_columns = ['gene', 'category', 'freq_allelic', 'variant_effect', 'clinvar', 'aminoacid_change']

class CNVModelView(ModelView):
    datamodel = SQLAInterface(Cnv)
    list_title = "CNV"
    show_title = "CNV"
    base_permissions = ['can_list', 'can_show']
    label_columns = {'gene': 'Genes', 'category': 'Variant Class', 'copy_number': 'Copy Number', }
    list_columns = ['gene', 'category', 'copy_number']

class ProjectFilesGeneralModelView(ModelView):
    datamodel = SQLAInterface(Wsifile)
    list_title = "WSI File List"
    show_title = "WSI File"
    base_permissions = ['can_list', 'can_show', 'can_edit']
    edit_exclude_columns = ['staining', 'type', 'annotationfile', 'patient', 'added_on', 'url_path']
    label_columns = {'annotate': 'WSI Name', 'annotationfile': 'Annotation Files', 'comments': 'Comments',
                     'tumor_cellularity': 'Tumor cellularity (%)'}
    list_columns = ['annotate', 'staining', 'type', 'tumor_cellularity', 'comments', 'annotationfile']
    related_views = [AnnotationFilesModelView]
    show_template = 'appbuilder/general/model/show_cascade.html'

class ProjectFilesTestGeneralModelView(ModelView):
    datamodel = SQLAInterface(WsifileTest)
    list_title = "WSI File List Test"
    show_title = "WSI File Test"
    base_permissions = ['can_list', 'can_show']
    label_columns = {'annotate': 'WSI Name',
                     'comments': 'Comments'}
    list_columns = ['annotate', 'comments']

class PatientModelView(ModelView):
    datamodel = SQLAInterface(Patient)

    list_title = "Patients"
    base_permissions = ['can_list', 'can_show', 'can_search']
    list_columns = ['ref', 'hospital.name', 'distant_metastases', 'status', 'clinical_stage']
    search_columns = ['ref', 'hospital', 'clinical_stage', 'wsifile']
    label_columns = {'ptnm': 'pTNM'}
    show_title = "Patient"

    show_fieldsets = [
        (
            'Summary',
            {'fields': ['ref', 'age_when_diagnosed', 'hospital',
                        'history', 'evolution', 'treatment', 'macroscopy',
                        'histological_classification', 'vessel_and_neural_invasion',
                        'ptnm', 'clinical_stage', 'immunohistochemistry'],
             'expanded': True}
        )
    ]
    related_views = [ProjectFilesModelView, VariantsModelView, CNVModelView]
    show_template = 'appbuilder/general/model/show_cascade.html'

class Protocol(BaseView):
    default_view = 'showProtocol'

    @expose('/showProtocol')
    @has_access
    def showProtocol(self):
        self.update_redirect()
        return self.render_template('protocol.html')

class DownloadTCGA(BaseView):
    default_view = 'downloadTCGA'

    @expose('/downloadTCGA')
    @has_access
    def downloadTCGA(self):
        self.update_redirect()
        return self.render_template('downloadTCGA.html')

    @expose('/download/<path:filename>', methods=['GET'])
    @has_access
    def download(self, filename):
        print(filename)
        return send_from_directory(directory=app.app.config['TCGA_DOWNLOAD_FOLDER'], filename=filename)


class Overview(BaseView):
    @expose('/getPatientOverview')
    @has_access
    def getPatientOverview(self):
        datamodel = SQLAInterface(Patient, )
        patients = db.session.query(Patient).all()
        data = []

        patient_evolution = {}
        patient_pattern = {}

        matrix_seleccion = np.zeros((2, 13))

        for patient in patients:
            for wsi in patient.wsifile:
                if wsi.staining.name == "HE" and wsi.type == "Surgical Specimen":
                    add_ = 1
                    for annotationfile in wsi.annotationfile:
                        if annotationfile.validated == True and annotationfile.annotationtype_id == 1:
                            add_ = 0
                    if add_ == 1:
                        row = 0
                        if patient.evolution.distant_metastases == True:
                            row = 1
                        matrix_seleccion[row, patient.histological_classification.IC_histological_type_id - 1] = \
                            matrix_seleccion[0, patient.histological_classification.IC_histological_type_id - 1] + 1
                        patient_evolution[patient.id] = patient.evolution.distant_metastases
                        patient_pattern[patient.id] = patient.histological_classification.IC_histological_type_id
                        break
        # print(matrix_seleccion )
        p = 0.2
        matrix_seleccion = np.ceil(matrix_seleccion * 0.2)
        # print(matrix_seleccion )

        for patient in patients:
            selected = ""
            for wsi in patient.wsifile:
                if wsi.staining.name == "HE" and wsi.type == "Surgical Specimen":
                    add_ = 1
                    for annotationfile in wsi.annotationfile:
                        if annotationfile.validated == True and annotationfile.annotationtype_id == 1:
                            add_ = 0
                    if add_ == 1:
                        row = 0
                        if patient.evolution.distant_metastases == True:
                            row = 1
                        if matrix_seleccion[row, patient.histological_classification.IC_histological_type_id - 1] > 0:
                            selected = "_shine"
                            matrix_seleccion[row, patient.histological_classification.IC_histological_type_id - 1] = \
                                matrix_seleccion[0, patient.histological_classification.IC_histological_type_id - 1] - 1
                        break

            p = {}

            if patient.hospital_id == 1:
                p["code"] = "Patient_HUPM" + selected
                hosp = " (HU Puerta del Mar)"
            elif patient.hospital_id == 2:
                p["code"] = "Patient_HUPR" + selected
                hosp = " (HU Puerto Real)"
            elif patient.hospital_id == 3:
                p["code"] = "Patient_HL" + selected
                hosp = " (H La Línea)"
            elif patient.hospital_id == 4:
                p["code"] = "Patient_HJF" + selected
                hosp = " (H Jerez de la Frontera)"

            p["name"] = patient.ref + hosp
            node = []
            nodeHE = {}
            nodeIMM = {}
            nodeHE["name"] = "HE"
            nodeIMM["name"] = "IMM"
            HE = []
            IMM = []
            count_HE = 0
            count_IMM = 0
            for wsi in patient.wsifile:
                leaf = {}
                leaf['name'] = wsi.url_path
                leaf['size'] = 1
                leaf['children'] = []
                if wsi.staining.name == "HE":
                    if len(wsi.annotationfile) > 0:
                        wsi_code = "HE_10"
                        for annotationfile in wsi.annotationfile:
                            if annotationfile.validated == True:
                                wsi_code = "HE_11"
                                break
                    else:
                        wsi_code = "HE_00"
                        count_HE += 1
                    leaf['code'] = wsi_code
                    HE.append(leaf)
                else:
                    if len(wsi.annotationfile) > 0:
                        wsi_code = "IMM_10"
                        for annotationfile in wsi.annotationfile:
                            if annotationfile.validated == True:
                                wsi_code = "IMM_11"
                                break
                    else:
                        wsi_code = "IMM_00"
                        count_IMM += 1
                    leaf['code'] = wsi_code
                    IMM.append(leaf)
            nodeHE["children"] = HE
            nodeIMM["children"] = IMM
            if len(HE) == 0:
                nodeHE["code"] = 0
            else:
                nodeHE["code"] = count_HE / len(HE)
            if len(IMM) == 0:
                nodeIMM["code"] = 0
            else:
                nodeIMM["code"] = count_IMM / len(IMM)
            node.append(nodeHE)
            node.append(nodeIMM)
            p["children"] = node
            data.append(p)
            root_ = {}
            root_["name"] = "Patients"
            root_["children"] = data
        return jsonify(root_)


appbuilder.add_view_no_menu(AnnotationFilesModelView)
appbuilder.add_view_no_menu(ProjectFilesModelView)
appbuilder.add_view_no_menu(Annotation)
appbuilder.add_view_no_menu(Overview)
appbuilder.add_view_no_menu(VariantsModelView)
appbuilder.add_view_no_menu(CNVModelView)

appbuilder.add_view(PatientModelView, "Patient List", icon="fa-address-book-o", category="Patients")
appbuilder.add_view(ProjectFilesGeneralModelView, "WSI File List", icon="fa fa-file-archive-o", category="WSI Files")
appbuilder.add_view(ProjectFilesTestGeneralModelView, "WSI File Test List", icon="fa fa-file-archive-o", category="Test Files")
appbuilder.add_view(DownloadTCGA, "Download TCGA Snapshots", icon="fa fa-file-archive-o", category="Test Files")
appbuilder.add_view(Protocol, "Annotation Protocol", icon="fa fa-file-text-o", category='Annotation Files')
appbuilder.add_view(AnnotationFilesGeneralModelView, "Annotation File List", icon="fa fa-file-archive-o",
                    category="Annotation Files")


appbuilder.security_cleanup()


@appbuilder.app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html', base_template=appbuilder.base_template, appbuilder=appbuilder), 404


db.create_all()
