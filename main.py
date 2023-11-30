import tensorflow as tf
import tensorflow_hub as hub
import cv2
import numpy as np

# Load MoveNet Thunder model
model = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
input_size = 256  # Thunder model's input size

def movenet(input_image):
    input_image = tf.cast(input_image, dtype=tf.int32)
    outputs = model.signatures['serving_default'](input_image)
    keypoints_with_scores = outputs['output_0'].numpy()
    return keypoints_with_scores

def draw_keypoints(frame, keypoints, confidence_threshold=0.4):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 5, (0, 255, 0), -1)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = tf.image.resize_with_pad(tf.expand_dims(frame, axis=0), input_size, input_size)
    keypoints_with_scores = movenet(img)
    draw_keypoints(frame, keypoints_with_scores)

    cv2.imshow('MoveNet Pose Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
