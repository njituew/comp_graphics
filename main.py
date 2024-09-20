from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import os

SMOOTHING_RADIUS = 50  # Радиус для сглаживания

app = Flask(__name__)

def moving_average(curve, radius):
    window_size = 2 * radius + 1
    f = np.ones(window_size) / window_size
    curve_pad = np.pad(curve, (radius, radius), 'edge')
    curve_smoothed = np.convolve(curve_pad, f, mode='same')
    return curve_smoothed[radius:-radius]

def smooth_trajectory(trajectory):
    smoothed_trajectory = np.copy(trajectory)
    for i in range(3):
        smoothed_trajectory[:, i] = moving_average(trajectory[:, i], SMOOTHING_RADIUS)
    return smoothed_trajectory

def stabilize_video(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    _, prev_frame = cap.read()
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    height, width = prev_gray.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, 30.0, (width, height))

    transforms = []
    
    while True:
        success, curr_frame = cap.read()
        if not success:
            break

        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=30, blockSize=3)
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)
        valid_prev_pts = prev_pts[status == 1]
        valid_curr_pts = curr_pts[status == 1]
        m = cv2.estimateAffinePartial2D(valid_prev_pts, valid_curr_pts)[0]

        dx = m[0, 2]
        dy = m[1, 2]
        da = np.arctan2(m[1, 0], m[0, 0])
        transforms.append([dx, dy, da])
        prev_gray = curr_gray.copy()

    transforms = np.array(transforms)
    trajectory = np.cumsum(transforms, axis=0)
    smoothed_trajectory = smooth_trajectory(trajectory)
    difference = smoothed_trajectory - trajectory
    transforms_smooth = transforms + difference

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    for i in range(len(transforms_smooth)):
        success, curr_frame = cap.read()
        if not success:
            break

        dx = transforms_smooth[i, 0]
        dy = transforms_smooth[i, 1]
        da = transforms_smooth[i, 2]

        m = np.array([[np.cos(da), -np.sin(da), dx],
                      [np.sin(da), np.cos(da), dy]])
        stabilized_frame = cv2.warpAffine(curr_frame, m, (width, height))
        stabilized_frame = stabilized_frame[20:-20, 20:-20]
        stabilized_frame = cv2.resize(stabilized_frame, (width, height))
        out.write(stabilized_frame)

    cap.release()
    out.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_video():
    video = request.files['video']
    input_video_path = os.path.join("uploads", video.filename)
    output_video_path = os.path.join("outputs", 'out_' + video.filename)
    video.save(input_video_path)

    stabilize_video(input_video_path, output_video_path)

    return send_file(output_video_path, as_attachment=True)

if __name__ == "__main__":
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    if not os.path.exists('outputs'):
        os.makedirs('outputs')
    app.run(debug=True, port=4200)
