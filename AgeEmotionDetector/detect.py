import cv2
from deepface import DeepFace

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    try:
        # Analyze frame
        result = DeepFace.analyze(frame, actions=['age', 'emotion'], enforce_detection=False)
        age = result[0]['age']
        emotion = result[0]['dominant_emotion']

        cv2.putText(frame, f"Age: {age}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        cv2.putText(frame, f"Emotion: {emotion}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    except:
        pass

    cv2.imshow("Age & Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
