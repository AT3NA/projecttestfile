from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import mediapipe as niggu

app = Flask(__name__)
socketio = SocketIO(app)

mp_hands = niggu.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = niggu.solutions.drawing_utils

def recognize_gesture(hand_landmarks, frame):
    h, w, _ = frame.shape
    landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
    fingertips = [8, 12, 16, 20]
    if all(landmarks[i][1] < landmarks[i - 2][1] for i in fingertips) and landmarks[4][0] < landmarks[3][0]:
        return 'B'
    elif (landmarks[8][1] < landmarks[6][1] and
          landmarks[12][1] < landmarks[10][1] and
          landmarks[16][1] < landmarks[14][1] and
          landmarks[20][1] < landmarks[18][1]):
        return 'A'
    return None

@socketio.on('connect')
def handle_connect():
    print('Client connected')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                gesture = recognize_gesture(hand_landmarks, frame)
                if gesture:
                    socketio.emit('gesture_result', {'gesture': gesture})
                    cv2.putText(frame, f'Alphabet is: {gesture}', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 60), 2)
        
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, debug=True)
