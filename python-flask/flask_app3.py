# 경로변수

from flask import Flask

app = Flask(__name__)

@app.route('/band/<band_id>') #http://127.0.0.1:5000/band/1234
def band1(band_id):

    return band_id

@app.route('/band/<band_id>/<band_menu>') #http://127.0.0.1:5000/band/1234/chat
def band2(band_id, band_menu):

    return band_id + ' ' + band_menu


app.run(host='127.0.0.1', port=5000, debug=False)


