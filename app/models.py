
import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, BigInteger, Date, Table, Sequence
from sqlalchemy.orm import relationship
from flask_appbuilder import Model
from sqlalchemy.orm import backref
from flask_appbuilder.models.decorators import renders

from flask import Markup, url_for

mindate = datetime.date(datetime.MINYEAR, 1, 1)

class Her2(Model):
    __tablename__ = 'her2'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Her2_ish(Model):
    __tablename__ = 'her2_ish'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Immunohistochemistry(Model):
    __tablename__ = 'immunohistochemistry'
    id = Column(Integer, primary_key=True)
    ER = Column(Float)
    PR = Column(Float)
    AR = Column(Float)
    p53 = Column(Float)
    Ki_67 = Column(Float)
    E_cadherin = Column(String)
    cytokeratin_19 = Column(Float)

    ER_project = Column(Float)
    PR_project = Column(Float)
    AR_project = Column(Float)
    p53_project = Column(Float)
    Ki_67_project = Column(Float)
    E_cadherin_project = Column(String)


    her2_id = Column(Integer, ForeignKey('her2.id'))
    her2 = relationship(Her2)
    her2_ish_id = Column(Integer, ForeignKey('her2_ish.id'))
    her2_ish = relationship(Her2_ish)
    her2_project = Column(String)



    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates="immunohistochemistry")

    def __repr__(self):
        out = ""
        if self.ER is not None:
            out = out + "ER: %.2f%%" % (self.ER*100)
        if self.ER_project is not None:
            out = out + "\nER (project): %.2f%%" % (self.ER_project*100)
        if self.PR is not None:
            out = out + "\nPR: %.2f%%" % (self.PR*100)
        if self.PR_project is not None:
            out = out + "\npR (project): %.2f%%" % (self.PR_project*100)
        if self.AR is not None:
            out = out + "\nAR: %.2f%%" % (self.AR*100)
        if self.AR_project is not None:
            out = out + "\nAR (project): %.2f%%" % (self.AR_project*100)
        if self.p53 is not None:
            out = out + "\np53: %.2f%%" % (self.p53*100)
        if self.p53_project is not None:
            out = out + "\np53 (project): %.2f%%" % (self.p53_project*100)
        if self.Ki_67 is not None:
            out = out + "\nKi 67: %.2f%%" % (self.Ki_67*100)
        if self.Ki_67_project is not None:
            out = out + "\nKi 67 (project): %.2f%%" % (self.Ki_67_project*100)
        if self.E_cadherin is not None:
            out = out + "\nE cadherin: %s" % self.E_cadherin
        if self.E_cadherin_project is not None:
            out = out + "\nE cadherin (project): %s" % self.E_cadherin_project
        if self.cytokeratin_19 is not None:
            out = out + "\nCytokeratin 19: %.2f%%" % (self.cytokeratin_19*100)
        if self.her2 is not None:
            out = out + "\nHer2: %s" % self.her2
        if self.her2_project is not None:
            out = out + "\nHer2 (project): %s" % self.her2_project
        if self.her2_ish is not None:
            out = out + "\nHer2-ISH: %s" % self.her2_ish

        return out

class Clinical_stage(Model):
    __tablename__ = 'clinical_stage'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Ptnm(Model):
    __tablename__ = 'ptnm'
    id = Column(Integer, primary_key=True)
    pT = Column(String(50))
    pN = Column(String(50))
    pM = Column(String(50))

    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates='ptnm')

    def __repr__(self):
        out = ""
        if self.pT is not None:
            out = "pT: %s" % (self.pT)
        if self.pN is not None:
            out = out + "\npN: %s" % (self.pN)
        if self.pM is not None:
            out = out + "\npM: %s" % (self.pM)
        return out

class Invasion_of_blood_lymph_perineural(Model):
    __tablename__ = 'invasion_of_blood_lymph_perineural'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Lymph_node_type_of_metastases(Model):
    __tablename__ = 'lymph_node_type_of_metastases'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Vessel_and_neural_invasion(Model):
    __tablename__ = 'vessel_and_neural_invasion'
    id = Column(Integer, primary_key=True)
    invasion_of_blood_lymph_perineural_id = Column(Integer, ForeignKey('invasion_of_blood_lymph_perineural.id'))
    invasion_of_blood_lymph_perineural = relationship(Invasion_of_blood_lymph_perineural)
    lymph_node_number = Column(Integer)
    lymph_node_maximum_size = Column(Integer)
    lymph_node_number_with_metastases = Column(Integer)
    lymph_node_extracapsular_extension = Column(Boolean)
    lymph_node_type_of_metastases_id = Column(Integer, ForeignKey('lymph_node_type_of_metastases.id'))
    lymph_node_type_of_metastases = relationship(Lymph_node_type_of_metastases)

    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates='vessel_and_neural_invasion')
    def __repr__(self):
        out = ""
        if self.invasion_of_blood_lymph_perineural is not None:
            out = "Invasion of blood/lymph/perineural: %s" % (self.invasion_of_blood_lymph_perineural)
        if self.lymph_node_number is not None:
            out = out + "\nDissected lymph node number: %s" % str(self.lymph_node_number)
        if self.lymph_node_maximum_size is not None:
            out = out + "\nLymph node maximum size (mm): %s" % str(self.lymph_node_maximum_size)
        if self.lymph_node_number_with_metastases is not None:
            out = out + "\nLymph node number with metastases: %s" % str(self.lymph_node_number_with_metastases)
        if self.lymph_node_extracapsular_extension is not None:
            out = out + "\nLymph node extracapsular extension: %s" % (self.lymph_node_extracapsular_extension)
        if self.lymph_node_type_of_metastases is not None:
            out = out + "\nLymph node type of metastases: %s" % (self.lymph_node_type_of_metastases)
        return out


class ISC_histological_type(Model):
    __tablename__ = 'ISC_histological_type'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class IC_histological_type(Model):
    __tablename__ = 'IC_histological_type'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Nearest_surgical_margin(Model):
    __tablename__ = 'nearest_surgical_margin'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Quadrant(Model):
    __tablename__ = 'quadrant'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Side(Model):
    __tablename__ = 'side'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Surgical_intervention(Model):
    __tablename__ = 'surgical_intervention'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Treatment(Model):
    __tablename__ = 'treatment'
    id = Column(Integer, primary_key=True)
    previous_neoadjuvance = Column(Boolean)

    neoadjuvance_type = Column(String)
    neoadjuvance_drug = Column(String)
    neoadjuvance_cycles = Column(String)


    surgical_intervention_id = Column(Integer, ForeignKey('surgical_intervention.id'))
    surgical_intervention = relationship(Surgical_intervention)

    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates='treatment')

    def __repr__(self):
        out =""
        if self.previous_neoadjuvance is not None:
            out = "Previous neoadjuvance: %s" % (self.previous_neoadjuvance)
        if self.neoadjuvance_type is not None:
            out = out + "\nNeoadjuvance type: %s" % self.neoadjuvance_type
        if self.neoadjuvance_drug is not None:
            out = out + "\nNeoadjuvance drug: %s" % self.neoadjuvance_drug
        if self.neoadjuvance_cycles is not None:
            out = out + "\nNeoadjuvance cycles: %s" % self.neoadjuvance_cycles
        if self.surgical_intervention is not None:
            out = out + "\nSurgical intervention: %s" % self.surgical_intervention
        return out

class Macroscopy(Model):
    __tablename__ = 'macroscopy'
    id = Column(Integer, primary_key=True)
    side_id = Column(Integer, ForeignKey('side.id'))
    side = relationship(Side)
    quadrant_id = Column(Integer, ForeignKey('quadrant.id'))
    quadrant = relationship(Quadrant)
    tumor_size = Column(Integer)
    distance_to_surgical_margin = Column(Integer)
    nearest_surgical_margin_id = Column(Integer, ForeignKey('nearest_surgical_margin.id'))
    nearest_surgical_margin = relationship(Nearest_surgical_margin)
    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates='macroscopy')

    def __repr__(self):
        out =""
        if self.side is not None:
            out = "Side: %s" % self.side
        if self.quadrant is not None:
            out = out + "\nQuadrant: %s" % self.quadrant
        if self.tumor_size is not None:
            out = out + "\nTumor size (mm): %s" % str(self.tumor_size)
        if self.distance_to_surgical_margin is not None:
            out = out + "\nDistance to surgical margin (mm): %s" % str(self.distance_to_surgical_margin)
        if self.nearest_surgical_margin is not None:
            out = out + "\nNearest surgical margin: %s" % str(self.nearest_surgical_margin)
        return out

class Hospital(Model):
    __tablename__ = 'hospital'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    def __repr__(self):
        return self.name

class Distant_metastases(Model):
    __tablename__ = 'distant_metastases'
    id = Column(Integer, primary_key=True)
    value = Column(Boolean, unique=True,  nullable=False)
    def __repr__(self):
        return str(self.value)

class Status(Model):
    __tablename__ = 'status'
    id = Column(Integer, primary_key=True)
    value = Column(String(50), unique=True)
    def __repr__(self):
        return self.value

class Evolution(Model):
    __tablename__ = 'evolution'
    id = Column(Integer, primary_key=True)
    distant_metastases = Column(Boolean)
    status_id = Column(Integer, ForeignKey('status.id'))
    status = relationship("Status")
    relapse = Column(Boolean)
    age_when_metastases_diagnosed = Column(Integer)

    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates='evolution')

    def __repr__(self):
        out =""
        if self.distant_metastases is not None:
            out = "Distant metastases: '%s'" % self.distant_metastases
        if self.age_when_metastases_diagnosed is not None:
            out = out + "\n Age when metastases was diagnosed: '%s'" % str(self.age_when_metastases_diagnosed)
        if self.status is not None:
            out = out + "\nStatus: '%s'" % self.status
        if self.relapse is not None:
            out = out + "\nRelapse: %s" % self.relapse
        return out

class History(Model):
    __tablename__ = 'history'
    id = Column(Integer, primary_key=True)
    personal_history = Column(Boolean)
    personal_history_cancer = Column(String(50))
    family_history = Column(Boolean)
    family_history_grade = Column(BigInteger)


    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates='history')

    def __repr__(self):
        out = ""
        if self.personal_history is not None:
            out = "Personal history: %s" % self.personal_history
        if self.personal_history_cancer is not None:
            out = out + "\nPersonal history cancer: %s" % self.personal_history_cancer
        if self.family_history is not None:
            out = out + "\nFamily history: %s" % self.family_history
        if self.family_history_grade is not None:
            grade_str = ' 1º: ' + str(bool(divmod(self.family_history_grade, 1000)))
            grade_str = grade_str + '; 2º: ' + str(bool(divmod(self.family_history_grade, 100)))
            grade_str = grade_str + '; 3º: ' + str(bool(divmod(self.family_history_grade, 10)))
            grade_str = grade_str + '; 4º: ' + str(bool(divmod(self.family_history_grade, 1)))
            out = out + "\nFamily history grade:" + grade_str

        return out



class Histological_classification(Model):
    __tablename__ = 'histological_classification'
    id = Column(Integer, primary_key=True)

    ISC_histological_type_id = Column(Integer, ForeignKey('ISC_histological_type.id'))
    ISC_histological_type = relationship(ISC_histological_type, foreign_keys=[ISC_histological_type_id])

    ISC_grade = Column(Integer)
    ISC_architectural_pattern = Column(String)

    IC_histological_type_id = Column(Integer, ForeignKey('IC_histological_type.id'))
    IC_histological_type = relationship(IC_histological_type, foreign_keys=[IC_histological_type_id])
    IC_grade = Column(Integer)
    HE_grade_histopathological_pattern = Column(Integer)

    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates='histological_classification')


    def __repr__(self):
        out = ""
        if self.ISC_histological_type is not None:
            out = "ISC histological type: %s" % (self.ISC_histological_type)
        if self.ISC_grade is not None:
            out = out + "\nISC grade: %s" % str(self.ISC_grade)
        if self.ISC_architectural_pattern is not None:
            out = out + "\nISC architectural pattern: %s" % (self.ISC_architectural_pattern)
        if self.IC_histological_type is not None:
            out = out + "\nIC histological type: %s" % (self.IC_histological_type)
        if self.IC_grade is not None:
            out = out + "\nIC grade: %s" % str(self.IC_grade)
        if self.HE_grade_histopathological_pattern is not None:
            out = out + "\nHE grade histopathological pattern: %s" % str(self.HE_grade_histopathological_pattern)
        return out

class Patient(Model):

    __tablename__ = 'patient'

    id = Column(Integer, primary_key=True)
    ref = Column(String, unique=True)
    age_when_diagnosed = Column(Integer)
    hospital_id = Column(Integer, ForeignKey('hospital.id'), nullable=False)

    hospital = relationship("Hospital", uselist=False)
    history = relationship("History", uselist=False, back_populates="patient")
    evolution = relationship("Evolution", uselist=False, back_populates="patient")
    treatment = relationship("Treatment", uselist=False, back_populates="patient")
    macroscopy = relationship("Macroscopy", uselist=False, back_populates="patient")

    histological_classification = relationship("Histological_classification", uselist=False, back_populates="patient")
    vessel_and_neural_invasion = relationship('Vessel_and_neural_invasion', uselist=False, back_populates="patient")
    ptnm = relationship('Ptnm', uselist=False, back_populates="patient")
    clinical_stage_id = Column(Integer, ForeignKey('clinical_stage.id'))
    clinical_stage = relationship('Clinical_stage', uselist=False)
    immunohistochemistry = relationship("Immunohistochemistry", uselist=False, back_populates="patient")
    wsifile = relationship('Wsifile', back_populates='patient')
    variant = relationship('Variant', back_populates='patient')
    cnv = relationship('Cnv', back_populates='patient')

    def __repr__(self):
        return str(self.ref)

    def distant_metastases(self):
        return self.evolution.distant_metastases
    def status(self):
        return self.evolution.status

class Staining(Model):
    __tablename__ = 'staining'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    def __repr__(self):
        return self.name
class Clinvar(Model):
    __tablename__ = 'clinvar'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    def __repr__(self):
        return self.name

class Cnv(Model):
    __tablename__ = 'cnv'
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates="cnv")
    specimen = Column(String(200))
    cellularity = Column(Float)
    category = Column(String(200))
    mapd = Column(Float)
    filter = Column(String(200))
    gene = Column(String(200))
    copy_number = Column(Float)
    p_value = Column(Float)
    confidence = Column(String(200))
    cyto_band = Column(String(200))

    def __repr__(self):
        return self.gene

class Variant(Model):
    __tablename__ = 'variant'
    id = Column(Integer, primary_key=True)
    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates="variant")
    specimen =Column(String(200))
    cellularity = Column(Integer)
    category = Column(String(200))
    gene = Column(String(200))
    exon = Column(Integer)
    freq_allelic = Column(Float)
    aminoacid_change = Column(String(200))
    coding = Column(Integer)
    coverage = Column(Integer)
    variant_effect = Column(String(200))
    clinvar_id = Column(Integer, ForeignKey('clinvar.id'), nullable=False)
    clinvar = relationship("Clinvar", uselist=False)
    p_value = Column(Float)

    def __repr__(self):
        return self.gene


class Wsifile(Model):

    __tablename__ = 'wsifile'

    id = Column(Integer, primary_key=True)
    url_path = Column(String(200), unique=False)
    patient_id = Column(Integer, ForeignKey('patient.id'))
    patient = relationship("Patient", back_populates="wsifile")

    comments = Column(String(1000))
    tumor_cellularity = Column(Integer)

    staining_id = Column(Integer, ForeignKey('staining.id'), nullable=False)
    staining = relationship("Staining", uselist=False)

    type = Column(String(200))

    added_on = Column(String(200))
    annotationfile = relationship('Annotationfile', back_populates='wsifile')

    def annotate(self):
        return Markup(
            '<a href="' + url_for('Annotation.slide', file_path=self.url_path, annotation_path="None") + '">' +  self.url_path + '</a>')

    def __repr__(self):
        return self.url_path


class WsifileTest(Model):

    __tablename__ = 'wsifiletest'

    id = Column(Integer, primary_key=True)
    url_path = Column(String(200), unique=False)

    staining_id = Column(Integer, ForeignKey('staining.id'), nullable=False)
    staining = relationship("Staining", uselist=False)


    comments = Column(String(1000))

    def annotate(self):
        return Markup(
            '<a href="' + url_for('Annotation.slide', file_path=self.url_path, annotation_path="None") + '">' +  self.url_path + '</a>')

    def __repr__(self):
        return self.url_path


class Annotationtype(Model):
    __tablename__ = 'annotationtype'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True)
    def __repr__(self):
        return self.name

class Annotationfile(Model):

    __tablename__ = 'annotationfile'
    id = Column(Integer, primary_key=True)
    url_path = Column(String(200))
    annotationtype_id = Column(Integer, ForeignKey('annotationtype.id'), nullable=False)
    annotationtype = relationship("Annotationtype", uselist=False)
    created_by = Column(String(200))
    modified_on = Column(String(200))
    modified_by = Column(String(200))
    wsifile_id = Column(Integer, ForeignKey('wsifile.id'))
    wsifile = relationship("Wsifile", back_populates="annotationfile")
    validated = Column(Boolean)

    def __repr__(self):
        return self.url_path

    def open_annotation(self):
        return Markup(
            '<a href="' + url_for('Annotation.slide', file_path=self.wsifile.url_path, annotation_path=self.url_path) + '">' + self.url_path + '</a>')
