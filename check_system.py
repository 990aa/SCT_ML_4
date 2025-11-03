import os
import sys


def check_file_exists(filename, description):
    """Check if a file exists and print status"""
    exists = os.path.exists(filename)
    status = "✓" if exists else "✗"
    print(f"  {status} {description}: {filename}")
    return exists


def main():
    print("\n" + "=" * 70)
    print("Hand Gesture Recognition - System Check")
    print("=" * 70 + "\n")
    
    print("Checking required files...")
    
    # Check model file
    model_ok = check_file_exists(
        "hand_gesture_lstm_model.h5",
        "Trained model"
    )
    
    # Check gesture mapping
    mapping_ok = check_file_exists(
        "gesture_mapping.json",
        "Gesture mapping"
    )
    
    # Check webcam tester
    tester_ok = check_file_exists(
        "webcam_gesture_tester.py",
        "Webcam tester script"
    )
    
    # Check notebook
    notebook_ok = check_file_exists(
        "hand_gesture_recognition.ipynb",
        "Training notebook"
    )
    
    print("\n" + "-" * 70)
    
    # Overall status
    all_ok = model_ok and mapping_ok and tester_ok
    
    if all_ok:
        print("\n✓ All required files found! System is ready.\n")
        print("Next steps:")
        print("  1. Run webcam tester: python webcam_gesture_tester.py")
        print("  2. Or see WEBCAM_TESTER_GUIDE.md for detailed instructions")
    else:
        print("\n✗ Some files are missing.\n")
        
        if not model_ok or not mapping_ok:
            print("Missing model or mapping files. Please:")
            print("  1. Open hand_gesture_recognition.ipynb")
            print("  2. Run all cells to train the model")
            print("  3. This will generate both required files")
        
        if not tester_ok:
            print("Missing webcam tester script.")
            print("  Please ensure webcam_gesture_tester.py is in the directory")
    
    print("\n" + "=" * 70)
    
    # Try to load and display gesture mapping if available
    if mapping_ok:
        try:
            import json
            with open("gesture_mapping.json", 'r') as f:
                mapping = json.load(f)
            
            print("\nGesture Mapping (use numbers 1-9 in webcam tester):")
            print("-" * 70)
            for idx, (gesture, gesture_id) in enumerate(sorted(mapping.items(), key=lambda x: x[1]), 1):
                if idx <= 9:
                    print(f"  [{idx}] {gesture}")
            print("=" * 70 + "\n")
        except Exception as e:
            print(f"\nNote: Could not read gesture mapping: {e}\n")
    
    # Check Python packages
    print("Checking Python packages...")
    packages = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'mediapipe': 'mediapipe',
        'tensorflow': 'tensorflow'
    }
    
    missing_packages = []
    for module, package in packages.items():
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (MISSING)")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: uv pip install " + " ".join(missing_packages))
    else:
        print("\n✓ All required packages installed!")
    
    print("\n" + "=" * 70 + "\n")
    
    return all_ok and len(missing_packages) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
