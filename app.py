import os
from flask import Flask , request 
import process

app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def getImage():
    if request.method == 'POST':
        file = request.files['files']
        results = process.ImagePath(file.filename)
        print(results)
        return str(results)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)