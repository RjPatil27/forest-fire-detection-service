from flask import Flask, request, Response, make_response, send_file, render_template
from flask_cors import CORS, cross_origin
from werkzeug.utils import secure_filename
import cv2
import os

import image_processing
import video_processing
import video_copy


VIDEO_PATH = os.path.sep.join(["video"])

app = Flask(__name__)
CORS(app)
cors = CORS(app, resource={
    r"/*":{
        "origins":"*"
    }
})

@cross_origin()
@app.route('/upload/image', methods=['POST'])
def upload_image():
    file = request.files['image']
    file_name = secure_filename(file.filename)
    output_img = image_processing.run_fire_and_smoke_detection_on_image(file.read())
    image_type = file.content_type.split('/')
    retval, buffer = cv2.imencode('.' + image_type[1], output_img)
    response = make_response(buffer.tobytes())
    response.headers.set('Content-Type', file.content_type)
    response.headers.set('Content-Disposition', 'attachment', filename=file_name)
    return response

@cross_origin()
@app.route('/upload/video', methods=['POST'])
def upload_video():
    file = request.files['video']
    file_name = secure_filename(file.filename)
    file.save(os.path.join(VIDEO_PATH, file_name))
    video_path = video_processing.run_fire_and_smoke_detection_on_video(file_name)
    return send_file(video_path, attachment_filename='result.avi', as_attachment=True)


@cross_origin()
@app.route('/get/ndvi', methods=['GET'])
def get_ndvi():
    return render_template('ndvi.html')

@cross_origin()
@app.route('/get/ndwi', methods=['GET'])
def get_ndwi():
    return render_template('ndwi.html')


if __name__ == "__main__":
    app.run(debug=True)
