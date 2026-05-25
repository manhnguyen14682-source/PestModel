from flask import Flask, render_template_string, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np
import os
import base64
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = YOLO("best.pt")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Detection web</title>

    <style>

        body{
            font-family: Arial;
            background: #f4f4f4;
            margin: 0;
            padding: 40px;
            text-align: center;
        }

        h1{
            margin-bottom: 30px;
        }

        .upload-box{
            width: 500px;
            margin: auto;
            padding: 40px;
            border: 3px dashed #999;
            border-radius: 12px;
            background: white;
            transition: 0.3s;
        }

        .upload-box.dragover{
            border-color: #007bff;
            background: #eef5ff;
        }

        input[type=file]{
            display: none;
        }

        button{
            margin-top: 20px;
            padding: 12px 24px;
            border: none;
            background: #007bff;
            color: white;
            border-radius: 8px;
            cursor: pointer;
            font-size: 16px;
            transition: 0.2s;
        }

        button:hover{
            background: #0056b3;
        }

        .result{
            margin-top: 25px;
            font-size: 24px;
            font-weight: bold;
        }

        .image-container{
            width: 100%;
            display: flex;
            justify-content: center;
            margin-top: 25px;
        }

        #result-image{
            display: none;
            max-width: 900px;
            width: auto;
            border-radius: 12px;
            border: 3px solid #ccc;
            box-shadow: 0 4px 15px rgba(0,0,0,0.15);
        }

        .loading{
            display: none;
            margin-top: 30px;
        }

        .spinner{
            width: 60px;
            height: 60px;
            border: 7px solid #ddd;
            border-top: 7px solid #007bff;
            border-radius: 50%;
            animation: spin 1s linear infinite;
            margin: auto;
        }

        .loading-text{
            margin-top: 15px;
            font-size: 18px;
            font-weight: bold;
            color: #333;
        }

        @keyframes spin{
            0%{
                transform: rotate(0deg);
            }

            100%{
                transform: rotate(360deg);
            }
        }

    </style>
</head>

<body>

    <h1>Pest Detection</h1>

    <div class="upload-box" id="drop-area">

        <p><b>Drop Image Here</b></p>
        <p>or</p>

        <button onclick="document.getElementById('fileElem').click()">
            Choose from device
        </button>

        <input type="file" id="fileElem" accept="image/*">

    </div>

    <div class="loading" id="loading">

        <div class="spinner"></div>

        <div class="loading-text">
            Loading...
        </div>

    </div>

    <div class="result" id="count"></div>

    <div class="image-container">

        <img id="result-image"/>

    </div>

<script>

const dropArea = document.getElementById("drop-area");
const fileInput = document.getElementById("fileElem");
const loading = document.getElementById("loading");

["dragenter", "dragover"].forEach(eventName => {

    dropArea.addEventListener(eventName, (e) => {

        e.preventDefault();
        dropArea.classList.add("dragover");

    });

});

["dragleave", "drop"].forEach(eventName => {

    dropArea.addEventListener(eventName, (e) => {

        e.preventDefault();
        dropArea.classList.remove("dragover");

    });

});

dropArea.addEventListener("drop", (e) => {

    const file = e.dataTransfer.files[0];
    uploadFile(file);

});

fileInput.addEventListener("change", () => {

    const file = fileInput.files[0];
    uploadFile(file);

});

function uploadFile(file){

    if(!file) return;

    loading.style.display = "block";

    document.getElementById("count").innerHTML = "";

    let img = document.getElementById("result-image");
    img.style.display = "none";

    let formData = new FormData();
    formData.append("file", file);

    fetch("/predict", {

        method: "POST",
        body: formData

    })
    .then(res => res.json())
    .then(data => {

        loading.style.display = "none";

        document.getElementById("count").innerHTML =
            "Detected Objects: " + data.count;

        img.src = "data:image/jpeg;base64," + data.image;

        img.style.display = "block";

    })
    .catch(err => {

        loading.style.display = "none";

        alert("An error occurred!");

        console.log(err);

    });

}

</script>

</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(HTML)

@app.route("/predict", methods=["POST"])
def predict():

    file = request.files["file"]

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    file.save(filepath)

    # Predict
    results = model(filepath)

    count = len(results[0].boxes)

    annotated = results[0].plot(
        boxes=True,
        labels=True,
        conf=True,
        line_width=1,
        font_size=0.35,
    )

    _, buffer = cv2.imencode(".jpg", annotated)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    return jsonify({
        "count": count,
        "image": img_base64
    })

if __name__ == "__main__":
    app.run(debug=True)