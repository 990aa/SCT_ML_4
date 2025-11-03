"""
Real-time Hand Gesture Recognition Testing Interface
- Live gesture prediction
- Confidence scores
- Accuracy tracking
- Error metrics
- Visual feedback
"""

import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from collections import deque
import time
from datetime import datetime


class HandPoseExtractor:
    """Extract hand landmarks using MediaPipe"""
    
    def __init__(self, static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.mp_hands = mp.solutions.hands

    def extract_landmarks(self, image):
        """Extract hand landmarks from image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        landmarks = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])

        return np.array(landmarks) if landmarks else np.zeros(63)

    def draw_landmarks(self, image):
        """Draw hand landmarks on image"""
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        return image, results.multi_hand_landmarks is not None

    def close(self):
        self.hands.close()


class GestureTestingInterface:
    """Comprehensive gesture testing interface with accuracy tracking"""
    
    def __init__(self, model_path, gesture_mapping, sequence_length=30):
        print("Initializing Gesture Testing Interface...")
        
        # Load model
        self.model = tf.keras.models.load_model(model_path)
        print(f"✓ Model loaded from {model_path}")
        
        # Setup gesture mapping
        self.gesture_mapping = gesture_mapping
        self.id_to_gesture = {v: k for k, v in gesture_mapping.items()}
        self.num_classes = len(gesture_mapping)
        print(f"✓ Gesture classes: {list(gesture_mapping.keys())}")
        
        # Sequence parameters
        self.sequence_length = sequence_length
        self.sequence = deque(maxlen=sequence_length)
        
        # Hand pose extractor
        self.extractor = HandPoseExtractor(static_image_mode=False)
        
        # Testing mode variables
        self.testing_mode = False
        self.current_test_gesture = None
        self.test_predictions = []
        
        # Statistics tracking
        self.prediction_history = deque(maxlen=100)
        self.confidence_history = deque(maxlen=100)
        self.fps_history = deque(maxlen=30)
        
        # Per-gesture statistics
        self.gesture_stats = {name: {'correct': 0, 'total': 0, 'confidences': []} 
                             for name in gesture_mapping.keys()}
        
        # UI colors
        self.color_good = (0, 255, 0)      # Green
        self.color_medium = (0, 165, 255)  # Orange
        self.color_bad = (0, 0, 255)       # Red
        self.color_text = (255, 255, 255)  # White
        self.color_bg = (0, 0, 0)          # Black
        
        print("✓ Interface initialized successfully!\n")
    
    def process_frame(self, frame):
        """Process a single frame and return prediction"""
        landmarks = self.extractor.extract_landmarks(frame)
        self.sequence.append(landmarks)
        
        if len(self.sequence) == self.sequence_length:
            sequence_array = np.array([list(self.sequence)])
            predictions = self.model.predict(sequence_array, verbose=0)[0]
            predicted_class = np.argmax(predictions)
            confidence = predictions[predicted_class]
            gesture_name = self.id_to_gesture[predicted_class]
            
            # Store prediction history
            self.prediction_history.append(gesture_name)
            self.confidence_history.append(confidence)
            
            return gesture_name, confidence, predictions
        
        return None, 0.0, None
    
    def get_confidence_color(self, confidence):
        """Get color based on confidence level"""
        if confidence > 0.8:
            return self.color_good
        elif confidence > 0.5:
            return self.color_medium
        else:
            return self.color_bad
    
    def draw_ui(self, frame, gesture, confidence, all_predictions, hand_detected, fps):
        """Draw comprehensive UI overlay"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        
        # Draw semi-transparent panels
        # Top panel - Main prediction
        cv2.rectangle(overlay, (0, 0), (w, 120), self.color_bg, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        if hand_detected:
            if gesture:
                # Main prediction
                color = self.get_confidence_color(confidence)
                cv2.putText(frame, f"Gesture: {gesture}", (10, 40),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
                cv2.putText(frame, f"Confidence: {confidence:.1%}", (10, 80),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Confidence bar
                bar_width = int(300 * confidence)
                cv2.rectangle(frame, (10, 90), (310, 110), (50, 50, 50), -1)
                cv2.rectangle(frame, (10, 90), (10 + bar_width, 110), color, -1)
                cv2.rectangle(frame, (10, 90), (310, 110), (255, 255, 255), 2)
            else:
                cv2.putText(frame, "Collecting frames...", (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.color_medium, 2)
        else:
            cv2.putText(frame, "No hand detected", (10, 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, self.color_bad, 2)
        
        # FPS counter
        cv2.putText(frame, f"FPS: {fps:.1f}", (w - 150, 40),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_text, 2)
        
        # Testing mode indicator
        if self.testing_mode and self.current_test_gesture:
            cv2.rectangle(frame, (w - 400, 60), (w - 10, 110), (0, 100, 200), -1)
            cv2.putText(frame, f"Testing: {self.current_test_gesture}", (w - 390, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_text, 2)
        
        # Right panel - All predictions
        if all_predictions is not None and gesture:
            panel_x = w - 350
            panel_y = 130
            
            # Draw prediction panel background
            cv2.rectangle(frame, (panel_x - 10, panel_y - 10), 
                         (w - 10, panel_y + 30 * self.num_classes + 20), 
                         self.color_bg, -1)
            cv2.rectangle(frame, (panel_x - 10, panel_y - 10), 
                         (w - 10, panel_y + 30 * self.num_classes + 20), 
                         self.color_text, 2)
            
            cv2.putText(frame, "All Predictions:", (panel_x, panel_y - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_text, 2)
            
            # Sort predictions by confidence
            pred_items = [(self.id_to_gesture[i], prob) for i, prob in enumerate(all_predictions)]
            pred_items.sort(key=lambda x: x[1], reverse=True)
            
            for idx, (pred_gesture, prob) in enumerate(pred_items[:10]):  # Top 10
                y = panel_y + idx * 30
                
                # Highlight current prediction
                if pred_gesture == gesture:
                    cv2.rectangle(frame, (panel_x - 5, y - 20), 
                                (w - 15, y + 5), self.color_good, 2)
                
                # Draw probability bar
                bar_w = int(200 * prob)
                color = self.get_confidence_color(prob)
                cv2.rectangle(frame, (panel_x + 120, y - 15), 
                            (panel_x + 120 + bar_w, y), color, -1)
                
                # Draw text
                text = f"{pred_gesture[:10]}: {prob:.1%}"
                cv2.putText(frame, text, (panel_x, y),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_text, 1)
        
        # Bottom panel - Statistics
        if len(self.gesture_stats) > 0:
            total_correct = sum(stats['correct'] for stats in self.gesture_stats.values())
            total_tests = sum(stats['total'] for stats in self.gesture_stats.values())
            
            if total_tests > 0:
                accuracy = total_correct / total_tests
                error_rate = 1 - accuracy
                
                # Statistics panel
                stats_y = h - 100
                cv2.rectangle(frame, (10, stats_y - 10), (500, h - 10), self.color_bg, -1)
                cv2.rectangle(frame, (10, stats_y - 10), (500, h - 10), self.color_text, 2)
                
                cv2.putText(frame, f"Overall Accuracy: {accuracy:.1%}", (20, stats_y + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_good, 2)
                cv2.putText(frame, f"Error Rate: {error_rate:.1%}", (20, stats_y + 45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_bad, 2)
                cv2.putText(frame, f"Total Tests: {total_tests}", (20, stats_y + 75),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.color_text, 2)
                
                # Avg confidence
                if self.confidence_history:
                    avg_conf = np.mean(list(self.confidence_history))
                    cv2.putText(frame, f"Avg Confidence: {avg_conf:.1%}", (280, stats_y + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color_medium, 2)
        
        # Instructions
        inst_y = h - 110
        cv2.putText(frame, "Controls: [1-9]=Test Gesture | [Space]=Record | [R]=Reset | [S]=Save | [Q]=Quit", 
                   (10, inst_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.color_medium, 1)
        
        return frame
    
    def record_test_result(self, predicted_gesture, true_gesture, confidence):
        """Record a test result for accuracy tracking"""
        is_correct = (predicted_gesture == true_gesture)
        
        self.gesture_stats[true_gesture]['total'] += 1
        self.gesture_stats[true_gesture]['confidences'].append(confidence)
        
        if is_correct:
            self.gesture_stats[true_gesture]['correct'] += 1
        
        return is_correct
    
    def save_statistics(self):
        """Save statistics to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"gesture_test_results_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("=" * 60 + "\n")
            f.write("Hand Gesture Recognition Testing Results\n")
            f.write("=" * 60 + "\n\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model: {self.model}\n\n")
            
            total_correct = sum(stats['correct'] for stats in self.gesture_stats.values())
            total_tests = sum(stats['total'] for stats in self.gesture_stats.values())
            
            if total_tests > 0:
                accuracy = total_correct / total_tests
                f.write(f"Overall Accuracy: {accuracy:.2%}\n")
                f.write(f"Error Rate: {1-accuracy:.2%}\n")
                f.write(f"Total Tests: {total_tests}\n")
                f.write(f"Total Correct: {total_correct}\n\n")
                
                f.write("Per-Gesture Statistics:\n")
                f.write("-" * 60 + "\n")
                
                for gesture, stats in sorted(self.gesture_stats.items()):
                    if stats['total'] > 0:
                        gest_acc = stats['correct'] / stats['total']
                        avg_conf = np.mean(stats['confidences']) if stats['confidences'] else 0
                        f.write(f"\n{gesture}:\n")
                        f.write(f"  Accuracy: {gest_acc:.2%} ({stats['correct']}/{stats['total']})\n")
                        f.write(f"  Average Confidence: {avg_conf:.2%}\n")
                        f.write(f"  Error Rate: {1-gest_acc:.2%}\n")
        
        print(f"\n✓ Statistics saved to: {filename}")
        return filename
    
    def reset_statistics(self):
        """Reset all statistics"""
        self.gesture_stats = {name: {'correct': 0, 'total': 0, 'confidences': []} 
                             for name in self.gesture_mapping.keys()}
        print("\n✓ Statistics reset")
    
    def print_instructions(self):
        """Print usage instructions"""
        print("\n" + "=" * 70)
        print("WEBCAM GESTURE TESTING INTERFACE")
        print("=" * 70)
        print("\nControls:")
        print("  [1-9] - Enter testing mode for gesture (press number for gesture)")
        print("  [Space] - Record current prediction as test result")
        print("  [R] - Reset all statistics")
        print("  [S] - Save statistics to file")
        print("  [Q] or [ESC] - Quit")
        print("\nGesture Mapping:")
        for idx, (gesture_id, gesture_name) in enumerate(sorted(self.gesture_mapping.items(), key=lambda x: x[1]), 1):
            if idx <= 9:
                print(f"  [{idx}] - {gesture_name}")
        print("\n" + "=" * 70 + "\n")
    
    def run(self):
        """Run the testing interface"""
        self.print_instructions()
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Could not open webcam")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        print("Starting webcam... Press 'Q' to quit.\n")
        
        # Create gesture number mapping
        gesture_list = sorted(self.gesture_mapping.items(), key=lambda x: x[1])
        number_to_gesture = {str(i+1): gesture[0] for i, gesture in enumerate(gesture_list[:9])}
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Flip for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Draw hand landmarks and check detection
            frame, hand_detected = self.extractor.draw_landmarks(frame)
            
            # Process frame for prediction
            gesture, confidence, all_predictions = self.process_frame(frame)
            
            # Draw UI
            frame = self.draw_ui(frame, gesture, confidence, all_predictions, hand_detected, fps)
            
            # Display frame
            cv2.imshow('Hand Gesture Testing Interface', frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            # Quit
            if key == ord('q') or key == 27:  # Q or ESC
                break
            
            # Number keys for testing mode
            elif chr(key) in number_to_gesture:
                self.testing_mode = True
                self.current_test_gesture = number_to_gesture[chr(key)]
                print(f"\n→ Testing mode: {self.current_test_gesture}")
            
            # Space to record result
            elif key == ord(' ') and self.testing_mode and gesture:
                is_correct = self.record_test_result(gesture, self.current_test_gesture, confidence)
                status = "✓ CORRECT" if is_correct else "✗ WRONG"
                print(f"{status}: Predicted={gesture}, True={self.current_test_gesture}, Conf={confidence:.1%}")
            
            # Reset statistics
            elif key == ord('r'):
                self.reset_statistics()
            
            # Save statistics
            elif key == ord('s'):
                self.save_statistics()
            
            # Exit testing mode
            elif key == ord('t'):
                self.testing_mode = False
                self.current_test_gesture = None
                print("\n→ Testing mode disabled")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.extractor.close()
        
        # Print final statistics
        self.print_final_statistics()
    
    def print_final_statistics(self):
        """Print final statistics summary"""
        print("\n" + "=" * 70)
        print("FINAL STATISTICS")
        print("=" * 70)
        
        total_correct = sum(stats['correct'] for stats in self.gesture_stats.values())
        total_tests = sum(stats['total'] for stats in self.gesture_stats.values())
        
        if total_tests > 0:
            accuracy = total_correct / total_tests
            print(f"\nOverall Accuracy: {accuracy:.2%}")
            print(f"Error Rate: {1-accuracy:.2%}")
            print(f"Total Tests: {total_tests}")
            print(f"Total Correct: {total_correct}\n")
            
            print("Per-Gesture Performance:")
            print("-" * 70)
            for gesture, stats in sorted(self.gesture_stats.items()):
                if stats['total'] > 0:
                    gest_acc = stats['correct'] / stats['total']
                    avg_conf = np.mean(stats['confidences'])
                    print(f"{gesture:15} | Accuracy: {gest_acc:6.1%} | "
                          f"Correct: {stats['correct']:3}/{stats['total']:3} | "
                          f"Avg Conf: {avg_conf:6.1%}")
        else:
            print("\nNo tests performed.")
        
        print("=" * 70 + "\n")


def main():
    """Main function"""
    import json
    
    # Configuration
    MODEL_PATH = 'hand_gesture_lstm_model.h5'
    GESTURE_MAPPING_PATH = 'gesture_mapping.json'
    SEQUENCE_LENGTH = 30
    
    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        print("Please train the model first by running the notebook.")
        return
    
    # Load gesture mapping from file
    if os.path.exists(GESTURE_MAPPING_PATH):
        with open(GESTURE_MAPPING_PATH, 'r') as f:
            GESTURE_MAPPING = json.load(f)
        print(f"✓ Gesture mapping loaded from {GESTURE_MAPPING_PATH}")
    else:
        print(f"Warning: {GESTURE_MAPPING_PATH} not found. Using default mapping.")
        # Default gesture mapping (fallback)
        GESTURE_MAPPING = {
            '01_palm': 0,
            '02_l': 1,
            '03_fist': 2,
            '04_fist_moved': 3,
            '05_thumb': 4,
            '06_index': 5,
            '07_ok': 6,
            '08_palm_moved': 7,
            '09_c': 8,
            '10_down': 9
        }
    
    # Create and run interface
    interface = GestureTestingInterface(MODEL_PATH, GESTURE_MAPPING, SEQUENCE_LENGTH)
    interface.run()


if __name__ == "__main__":
    main()
