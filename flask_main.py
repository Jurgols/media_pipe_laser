from flask import Flask, render_template, Response, stream_with_context
from mediapipe_hands import CameraFetch



app = Flask(__name__)
@app.route('/')
def index():
    # rendering webpage
    return render_template('index.html')
def gen(camera):
    while True:
        #get camera frame
        frame = camera.mp_byte_array()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
@app.route('/video_feed')
def video_feed():
    cv_camera = CameraFetch(width=1280,height=720)
    return Response(stream_with_context(gen(cv_camera)),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
if __name__ == '__main__':
    # defining server ip address and port
    app.run(host="0.0.0.0", debug=True)