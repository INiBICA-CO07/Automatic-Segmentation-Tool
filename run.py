from app import app
from waitress import serve
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

serve(app, listen='0.0.0.0:443')
#app.run(host='0.0.0.0', port=8080, debug=True, use_reloader=False,  threaded=True,)
#if __name__ == "__main__":
#    app.run(host='0.0.0.0', port=8080, use_reloader=False,  threaded=True)

