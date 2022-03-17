from flask_appbuilder import IndexView, BaseView

class CustomIndexView(IndexView):
    index_template = 'wheel.html'