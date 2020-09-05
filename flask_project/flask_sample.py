from flask import Flask

app=Flask(__name__)

@app.route('/sample')
def running():
    return "<h1>HIHIHIHIHIHIHIHIHI</h1>"

@app.route('/predict')
def running2():
    return 'DOGGGGGGGG'