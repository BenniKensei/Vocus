"""Interactive gesture data collection and linear SVM training.

The script captures normalized landmark-distance vectors from a webcam, lets
the operator label each sample manually, and writes the resulting classifier to
``models/gesture_svm.pkl``.
"""

from __future__ import annotations

import os
import pickle

import cv2
import mediapipe as mp
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def extract_landmark_features(hand_landmarks) -> np.ndarray:
    """Convert MediaPipe hand landmarks into a normalized feature vector.

    Args:
        hand_landmarks: MediaPipe hand landmark object for a single hand.

    Returns:
        A 1D NumPy array with 20 normalized wrist-relative distances.

    Raises:
        None.
    """
    # Expected shape: (20,)
    wrist = hand_landmarks.landmark[0]
    features = []
    for i in range(1, 21):
        item = hand_landmarks.landmark[i]
        dist = ((item.x - wrist.x) ** 2 + (item.y - wrist.y) ** 2 + (item.z - wrist.z) ** 2) ** 0.5
        features.append(dist)

    # Expected shape: (20,)
    max_val = max(features) if features else 1.0
    if max_val > 0.0:
        features = [f / max_val for f in features]

    return np.array(features)


def main() -> None:
    """Run the interactive training loop and persist the fitted SVM.

    Args:
        None.

    Returns:
        None.

    Raises:
        None.
    """

    # TODO: Split collection and training into separate CLI commands once the
    # labeled dataset format is standardized.

    # Initialize MediaPipe Hands.
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)

    # Data stores.
    X_data = []
    y_data = []

    # Label trackers.
    class_counts = {0: 0, 1: 0, 2: 0}
    class_names = {0: "Up (u)", 1: "Down (d)", 2: "Neutral (n)"}

    print("--- Gesture SVM Training Pipeline ---")
    print("Hold gesture and press [u] for Up, [d] for Down, [n] for Neutral.")
    print("Press [t] to train and save the model.")
    print("Press [q] to abort process.")

    while True:
        ret, frame = cap.read()
        if not ret:
            continue
            
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        results = hands.process(rgb_frame)
        
        normalized_features = None

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            normalized_features = extract_landmark_features(hand_landmarks)

            # FIXME: Persist raw labeled samples if you need later auditing or
            # cross-validation beyond the current one-session workflow.
            
        # Draw status overlay
        y_offset = 30
        for class_id, count in class_counts.items():
            text = f"{class_names[class_id]}: {count}"
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30

        cv2.imshow("Gesture Data Collection", frame)

        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Process aborted by user.")
            break
            
        if normalized_features is not None:
            if key == ord('u'):
                X_data.append(normalized_features)
                y_data.append(0)
                class_counts[0] += 1
            elif key == ord('d'):
                X_data.append(normalized_features)
                y_data.append(1)
                class_counts[1] += 1
            elif key == ord('n'):
                X_data.append(normalized_features)
                y_data.append(2)
                class_counts[2] += 1
                
        if key == ord('t'):
            if not X_data:
                print("Error: No data blocks recorded. Cannot instantiate classifier.")
                continue
                
            print("\nInitiating linear Support Vector Machine training sequence...")
            
            # Expected shape: (num_samples, 20)
            X_array = np.array(X_data)
            # Expected shape: (num_samples,)
            y_array = np.array(y_data)
            
            # Train model. A linear kernel stays fast and interpretable for the
            # low-dimensional landmark-distance feature space.
            svm_model = SVC(kernel='linear', C=1.0)
            svm_model.fit(X_array, y_array)
            
            # Evaluate on the training set only. This is a sanity check, not a
            # generalization estimate.
            y_pred = svm_model.predict(X_array)
            acc = accuracy_score(y_array, y_pred)
            print(f"Training discrete accuracy on matrix: {acc * 100:.2f}%")
            
            # Subdir management and persistence.
            models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
            os.makedirs(models_dir, exist_ok=True)
            
            model_path = os.path.join(models_dir, "gesture_svm.pkl")
            
            with open(model_path, 'wb') as f:
                pickle.dump(svm_model, f)
                
            print(f"Structural integrity verified. Model serialized output >> {model_path}")
            break

    # Resource deallocation
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
