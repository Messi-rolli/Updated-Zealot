# Main pipeline runner script
import argparse

def main():
    parser = argparse.ArgumentParser(description="Distracted AI Driver Detection Pipeline")
    parser.add_argument('--mode', type=str, default='eye', choices=['eye', 'hand', 'lips', 'phone', 'all'],
                        help='Detection mode')
    args = parser.parse_args()
    if args.mode == 'eye':
        from drowsiness_detection.detectors.eye import run_eye_detection
        run_eye_detection()
    elif args.mode == 'hand':
        from drowsiness_detection.detectors.hand import run_hand_detection
        run_hand_detection()
    elif args.mode == 'lips':
        from drowsiness_detection.detectors.lips import run_lip_detection
        run_lip_detection()
    elif args.mode == 'phone':
        from drowsiness_detection.detectors.phone import run_phone_detection
        run_phone_detection()
    elif args.mode == 'all':
        import threading
        threads = []
        from drowsiness_detection.detectors.eye import run_eye_detection
        from drowsiness_detection.detectors.hand import run_hand_detection
        from drowsiness_detection.detectors.lips import run_lip_detection
        from drowsiness_detection.detectors.phone import run_phone_detection

        for func in [run_eye_detection, run_hand_detection, run_lip_detection, run_phone_detection]:
            t = threading.Thread(target=func)
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

if __name__ == "__main__":
    main()
