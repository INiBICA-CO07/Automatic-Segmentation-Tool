import logging
from flask import Flask
from flask_appbuilder import SQLA, AppBuilder
from .indexview import CustomIndexView
import os
from threading import Lock
import openslide
from openslide import OpenSlide, ImageSlide, open_slide, OpenSlideError
from openslide.deepzoom import DeepZoomGenerator
from collections import OrderedDict



logging.basicConfig(format='%(asctime)s:%(levelname)s:%(name)s:%(message)s')
logging.getLogger().setLevel(logging.DEBUG)

app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_object('config')
app.config.from_envvar('DEEPZOOM_MULTISERVER_SETTINGS', silent=True)


db = SQLA(app)


appbuilder = AppBuilder(app, db.session, indexview=CustomIndexView)

class _SlideCache(object):
    def __init__(self, cache_size, dz_opts):
        self.cache_size = cache_size
        self.dz_opts = dz_opts
        self._lock = Lock()
        self._cache = OrderedDict()

    def get(self, path):
        with self._lock:
            if path in self._cache:
                # Move to end of LRU
                slide = self._cache.pop(path)
                self._cache[path] = slide
                return slide

        osr = OpenSlide(path)
        slide = DeepZoomGenerator(osr, **self.dz_opts)
        try:
            mpp_x = osr.properties[openslide.PROPERTY_NAME_MPP_X]
            mpp_y = osr.properties[openslide.PROPERTY_NAME_MPP_Y]
            slide.mpp = (float(mpp_x) + float(mpp_y)) / 2
        except (KeyError, ValueError):
            slide.mpp = 0

        with self._lock:
            if path not in self._cache:
                if len(self._cache) == self.cache_size:
                    self._cache.popitem(last=False)
                self._cache[path] = slide
        return slide


app.basedir = os.path.abspath(app.config['SLIDE_DIR'])
app.appdir = os.path.abspath(app.config['APP_DIR'])
app.annotationdir = os.path.abspath(app.config['ANNOTATIONS_DIR'])
app.slidedictionaries = os.path.abspath(app.config['SLIDE_DICTIONARIES'])
app.deeplabmodelfile = os.path.abspath(app.config['DEEPLABMODEL_FILE'])
app.deeplabmodelfilecrf = os.path.abspath(app.config['DEEPLABMODELCRF_FILE'])
app.processinghost = app.config['PROCESSING_HOST']


config_map = {
    'DEEPZOOM_TILE_SIZE': 'tile_size',
    'DEEPZOOM_OVERLAP': 'overlap',
    'DEEPZOOM_LIMIT_BOUNDS': 'limit_bounds',
}
opts = dict((v, app.config[k]) for k, v in config_map.items())
app.cache = _SlideCache(app.config['SLIDE_CACHE_SIZE'], opts)


from app import views


