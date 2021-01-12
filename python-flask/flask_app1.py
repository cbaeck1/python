from flask import Flask
from flask import request

labels = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

app = Flask(__name__)

@app.route('/') #http://127.0.0.1:5000/
def index():
    html = '''
<html>
<head>
    <title>붗꽃 종류 예측</title>
</head>
<body>
    <center>
    붗꽃 종류 예측<br>
    <form action="/predict">
        꽃받침 길이 (cm) <input type="text" name="SepalLengthCm" value="5.3"><br>
        꽃받침 너비 (cm) <input type="text" name="SepalWidthCm" value="3.7"><br>
        꽃잎 길이 (cm) <input type="text" name="PetalLengthCm" value="1.5"><br>
        꽃잎 너비 (cm) <input type="text" name="PetalWidthCm" value="0.2"><br>
        <input type="submit" value="예측하기">
    </form>
    </center>
</body>
</html>
'''

    return html

@app.route('/predict') #http://127.0.0.1:5000/predict
def predict():
    SepalLengthCm = request.args.get('SepalLengthCm')
    SepalWidthCm = request.args.get('SepalWidthCm')
    PetalLengthCm = request.args.get('PetalLengthCm')
    PetalWidthCm = request.args.get('PetalWidthCm')

    print(SepalLengthCm)
    print(SepalWidthCm)
    print(PetalLengthCm)
    print(PetalWidthCm)

    y_predict = 0

    return labels[y_predict]

app.run(host='127.0.0.1', port=5000, debug=False)