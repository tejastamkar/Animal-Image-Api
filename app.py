import os
from flask import Flask, request
from process import imagePath

app = Flask(__name__)


@app.route("/detect", methods=["POST"])
def getImage():
    if request.method == 'POST':
        # file = request.files['files']
        imageFile = request.files['file']
        imageFile.save("temp.png")
        results = imagePath("temp.png")
        os.remove("temp.png")
        return str(results)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
