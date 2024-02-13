from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import math

app = Flask(__name__,template_folder='template')
cap = None
calculated_values = {}
height = 0

def pose_estimation():
    global cap
    with app.app_context():
        mp_pose = mp.solutions.pose
        pose = mp_pose.Pose()
        mp_drawing = mp.solutions.drawing_utils
        while True:
            if cap is not None:
                ret, frame = cap.read()
                if not ret:
                    break

                # Convert the frame to RGB format
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Perform pose estimation on the frame
                results = pose.process(frame_rgb)

                # Annotate the frame with pose estimation
                frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                if results.pose_landmarks:
                    landmarks = results.pose_landmarks.landmark
                    mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                              )
#----------------------------------------------------------------------------------------------------------------------------------------------------
                    def dist_bw_two_points(a,b,frame):
                        a_marks = landmarks[a]
                        b_marks = landmarks[b]

                        a_x, a_y = int(a_marks.x * frame.shape[1]), int(a_marks.y * frame.shape[0])
                        b_x, b_y = int(b_marks.x * frame.shape[1]), int(b_marks.y * frame.shape[0])

                        distance = math.sqrt((b_x - a_x) ** 2 + (b_y - a_y) ** 2)
                        cv2.circle(frame, (a_x, a_y), 5, (0, 255, 0), -1)
                        cv2.circle(frame, (b_x, b_y), 5, (0, 255, 0), -1)
                        
                        return distance
#----------------------------------------------------------------------------------------------------------------------------------------------------                  
                    # Relative Positioning of LEFT_WRIST AND RIGHT_WRIST
                    fixed_distance = 80/dist_bw_two_points(mp_pose.PoseLandmark.RIGHT_WRIST,mp_pose.PoseLandmark.LEFT_WRIST,frame)
                    
                    # LEFT ARM LENGTH
                    left_arm_length = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.LEFT_SHOULDER,mp_pose.PoseLandmark.LEFT_WRIST,frame)
                    
                    #RIGHT ARM LENGTH
                    right_arm_length = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.RIGHT_SHOULDER,mp_pose.PoseLandmark.RIGHT_WRIST,frame)

                    # LEFT LEG LENGTH
                    left_thigh_length = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.LEFT_HIP,mp_pose.PoseLandmark.LEFT_KNEE,frame)
                    left_knee_length = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.LEFT_FOOT_INDEX,mp_pose.PoseLandmark.LEFT_KNEE,frame)
                    
                    # RIGHT LEG LENGTH
                    right_thigh_length = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.RIGHT_HIP,mp_pose.PoseLandmark.RIGHT_KNEE,frame)
                    right_knee_length = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.RIGHT_FOOT_INDEX,mp_pose.PoseLandmark.RIGHT_KNEE,frame)
                    
                    # SHOULDER LENGTH
                    shoulder_length = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.RIGHT_SHOULDER,mp_pose.PoseLandmark.LEFT_SHOULDER,frame)
                    chest_circumference = 3.14*shoulder_length
                    
                    # WAIST CIRCUMFERENCE
                    waist_length = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.RIGHT_HIP,mp_pose.PoseLandmark.LEFT_HIP,frame)
                    waist_circumference = 3.14*waist_length
                    
                    # HEIGHT
                    height = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.NOSE,mp_pose.PoseLandmark.LEFT_HEEL,
                                                            frame) + 2*fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.NOSE,
                                                                                                        mp_pose.PoseLandmark.LEFT_EYE_INNER,frame)
                    #SITTING HEIGHT
                    sitting_height = height - (left_thigh_length+right_thigh_length)/2
                    
                    #FOOT LENGTH
                    foot_length = fixed_distance*dist_bw_two_points(mp_pose.PoseLandmark.LEFT_FOOT_INDEX,mp_pose.PoseLandmark.LEFT_HEEL,frame)
                    
                    
                    calculated_values['fixed_distance'] = str(round(80,2)) + " cm"
                    calculated_values['height'] = str(round(height,2)) + " cm"
                    calculated_values['left_arm_length'] = str(round(left_arm_length,2)) + " cm",
                    calculated_values['left_thigh_length'] = str(round(left_thigh_length,2)) + " cm",
                    calculated_values['left_knee_length'] = str(round(left_knee_length,2)) + " cm",
                    calculated_values['right_arm_length'] = str(round(right_arm_length,2)) + " cm",
                    calculated_values['right_thigh_length'] = str(round(right_thigh_length,2)) + " cm",
                    calculated_values['right_knee_length'] = str(round(right_knee_length,2)) + " cm",
                    calculated_values['shoulder_length'] = str(round(shoulder_length,2)) + " cm",
                    calculated_values['chest_circumference'] = str(round(chest_circumference,2)) + " cm",
                    calculated_values['waist_length'] = str(round(waist_length,2)) + " cm",
                    calculated_values['waist_circumference'] = str(round(waist_circumference,2)) + " cm",
                    calculated_values['sitting_height'] = str(round(sitting_height,2)) + " cm",
                    calculated_values['foot_length'] = str(round(foot_length,2)) + " cm"
                    


                # Convert the annotated frame to JPEG format
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()

                # Yield the frame as a response to the client
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(pose_estimation(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/start_video', methods=['POST'])
def start_video():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return 'OK'


@app.route('/stop_video', methods=['POST'])
def stop_video():
    global cap
    if cap is not None:
        cap.release()
        cap = None
        #print(calculated_values)
    return jsonify(calculated_values)


if __name__ == '__main__':
    app.run()
