import sys
import argparse
from jetson_inference import detectNet
from jetson_utils import videoSource, Log

# --- Configuration ---
MODEL_PATH = "/home/nvidia6/AIRCAT/detection/ssd/models/birddataset/ssd-mobilenet.onnx"
LABELS_PATH = "/home/nvidia6/AIRCAT/detection/ssd/models/birddataset/labels.txt"
CAMERA_URI = "/dev/video0" # Or "csi://0" for CSI camera
CONFIDENCE_THRESHOLD = 0.8 # Adjust as needed

# --- Setup Argument Parser (minimal) ---
parser = argparse.ArgumentParser(description="Prints detected objects from a live camera feed to the terminal.",
                                 formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument("--model", type=str, default=MODEL_PATH, help="Path to ONNX model")
parser.add_argument("--labels", type=str, default=LABELS_PATH, help="Path to class labels file")
parser.add_argument("--camera", type=str, default=CAMERA_URI, help="URI of the input camera stream")
parser.add_argument("--threshold", type=float, default=CONFIDENCE_THRESHOLD, help="Minimum detection threshold")

# Add arguments for ONNX input/output blobs, these usually stay consistent
parser.add_argument("--input-blob", type=str, default="input_0", help="Name of the input layer blob")
parser.add_argument("--output-cvg", type=str, default="scores", help="Name of the coverage layer blob")
parser.add_argument("--output-bbox", type=str, default="boxes", help="Name of the bounding box layer blob")

try:
    args = parser.parse_known_args()[0]
except:
    print("")
    parser.print_help()
    sys.exit(0)

# --- Load the object detection network ---
# Note: arguments passed directly to detectNet constructor for clarity
net = detectNet(model=args.model,
                labels=args.labels,
                input_blob=args.input_blob,
                output_cvg=args.output_cvg,
                output_bbox=args.output_bbox,
                threshold=args.threshold)

if net is None:
    Log.Error("detectNet failed to initialize. Please check model, labels, and input/output blob names.")
    sys.exit(1)

# --- Create video source ---
camera = videoSource(args.camera)

if camera is None:
    Log.Error(f"Failed to open video source {args.camera}")
    sys.exit(1)

Log.Info(f"Starting detection from camera: {args.camera}")
Log.Info(f"Model: {args.model}")
Log.Info(f"Labels: {args.labels}")
Log.Info(f"Threshold: {args.threshold}")

# --- Main loop for processing frames ---
try:
    while True:
        # Capture the next image frame
        img = camera.Capture()

        if img is None: # Timeout or EOS
            continue

        # Detect objects in the image
        # We don't use overlay here as we're not rendering to screen
        detections = net.Detect(img, overlay='none') # 'none' means no drawing is done internally

        # Print the detections to the terminal
        if len(detections) > 0:
            print(f"\nALERT! BIRD DETECTED! TAKE EVASIVE ACTION WHEN POSSIBLE!")
            for detection in detections:
                # Get the class label string
                class_label = net.GetClassDesc(detection.ClassID)

                print(f"  Label: {class_label}")
                print(f"  Confidence: {detection.Confidence:.2f}")
                print(f"  Bounding Box: Left={detection.Left:.2f}, Top={detection.Top:.2f}, Right={detection.Right:.2f}, Bottom={detection.Bottom:.2f}")
                # You can add more details from the 'detection' object if needed:
                # print(f"  Width: {detection.Width:.2f}, Height: {detection.Height:.2f}")
                # print(f"  Center: ({detection.Center[0]:.2f}, {detection.Center[1]:.2f})")
                print("--------------------")
        else:
            print(f"\nNo threats detected.")
            # Optionally print if no objects detected to show it's still running
            # print("No objects detected.")
            pass # Keep it quiet if nothing is found

        # Break loop if camera stream stops (though for live camera, it usually runs indefinitely)
        if not camera.IsStreaming():
            break

except Exception as e:
    Log.Error(f"An error occurred: {e}")
except KeyboardInterrupt:
    Log.Info("Detection process interrupted by user (Ctrl+C). Exiting.")
finally:
    Log.Info("Detection script finished.")