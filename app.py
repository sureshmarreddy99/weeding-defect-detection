import os
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
from model_logic import process_image

app = Flask(__name__)

UPLOAD_FOLDER = "static/uploads"
OUTPUT_FOLDER = "static/outputs"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "bmp"}

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET", "POST"])
def home():
    input_image_url = None
    output_image_url = None
    error = None
    result = None

    os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
    os.makedirs(app.config["OUTPUT_FOLDER"], exist_ok=True)

    if request.method == "POST":
        if "image" not in request.files:
            error = "No image file was uploaded."
            return render_template("index.html", error=error)

        file = request.files["image"]

        if file.filename == "":
            error = "Please select an image."
            return render_template("index.html", error=error)

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            input_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            output_filename = f"output_{filename}"
            output_path = os.path.join(app.config["OUTPUT_FOLDER"], output_filename)

            file.save(input_path)

            try:
                result = process_image(input_path, output_path)
                input_image_url = url_for("static", filename=f"uploads/{filename}")
                output_image_url = url_for("static", filename=f"outputs/{output_filename}")
            except Exception as e:
                error = str(e)
        else:
            error = "Unsupported file type. Please upload png, jpg, jpeg, or bmp."

    return render_template(
        "index.html",
        input_image_url=input_image_url,
        output_image_url=output_image_url,
        error=error,
        result=result
    )


if __name__ == "__main__":
    app.run(debug=True)
