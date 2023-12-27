import cv2
import numpy as np
import re
import json
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import subprocess
import zipfile
import argparse
from tqdm import tqdm

# Constants.
INPUT_WIDTH = 640
INPUT_HEIGHT = 640
SCORE_THRESHOLD = 0.5
NMS_THRESHOLD = 0.45
CONFIDENCE_THRESHOLD = 0.45


# Text parameters.
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors
BLACK  = (0,0,0)
BLUE   = (255,178,50)
YELLOW = (0,255,255)
RED = (0,0,255)


def draw_label(input_image, label, left, top):
    """Draw text onto image at location."""
    
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle. 
    cv2.rectangle(input_image, (left, top), (left + dim[0], top + dim[1] + baseline), BLACK, cv2.FILLED);
    # Display text inside the rectangle.
    cv2.putText(input_image, label, (left, top + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def format_yolov5(source):
    row, col = source.shape[:2]
    _max = max(col, row)
    # Create a blank square image with the same type as the source image
    result = np.zeros((_max, _max, 3), dtype=source.dtype)
    # Copy the original image to the top-left corner of the result image
    result[0:row, 0:col] = source
    return result

def pre_process(input_image, net):
    input_image=format_yolov5(input_image)
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1/255, (INPUT_WIDTH, INPUT_HEIGHT), [0,0,0], 1, crop=False)

    # Sets the input to the network.
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers.
    output_layers = net.getUnconnectedOutLayersNames()
    outputs = net.forward(output_layers)

    return outputs


def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    detections = []

    # Rows.
    rows = outputs[0].shape[1]

    image_height, image_width = input_image.shape[:2]
    _max=_max = max(image_width, image_height)

    # Resizing factor.
    x_factor = _max / INPUT_WIDTH
    y_factor =  _max / INPUT_HEIGHT

    # Iterate through 25200 detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]

        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]

            # Get the index of max class score.
            class_id = np.argmax(classes_scores)

            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)

                cx, cy, w, h = row[0], row[1], row[2], row[3]

                left = int((cx - w/2) * x_factor)
                top = int((cy - h/2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
              
                box = np.array([left, top, width, height])
                boxes.append(box)
                # Add a dictionary for each detection
                detections.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": box
                })

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    filtered_detections = []
    for i in indices:
        filtered_detections.append(detections[i])
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        label = "{}:{:.2f}".format(classes[class_ids[i]], confidences[i])
        draw_label(input_image, label, left, top)

    return input_image,filtered_detections

def download_with_wget_and_unzip(url, extract_to):
    # Download the file using wget
    zip_path = os.path.join(extract_to, 'val2017.zip')
    subprocess.run(['wget', '-O', zip_path, url])
    # Unzip the file
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    # Remove the zip file after extraction
    os.remove(zip_path)


def generate_coco_format(detections, image_id):
    coco_detections = []
    for det in detections:
        # print("Current detection before conversion:", det)

        # Convert NumPy types to Python types more robustly
        class_id = int(det['class_id']) if np.issubdtype(type(det['class_id']), np.integer) else det['class_id']
        confidence = float(det['confidence']) if np.issubdtype(type(det['confidence']), np.floating) else det['confidence']
        bbox = det.get('bbox', [0, 0, 0, 0])

        # Ensure bbox elements are Python integers
        bbox = [int(coord) if np.issubdtype(type(coord), np.integer) else coord for coord in bbox]

        coco_det = {
            "image_id": image_id,
            "category_id": COCO91_MAP[class_id],
            "bbox": bbox,
            "score": confidence
        }
        #print("Current detection after conversion:", coco_det)
        coco_detections.append(coco_det)

    return coco_detections


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='YOLOV5 inference for COCO')
    parser.add_argument('--input_images_folder', type=str, default="coco/val2017", help='input images path')
    parser.add_argument('--output_images_folder', type=str, default="coco/coco_output_py", help='output images path')
    parser.add_argument('--model_weights', type=str, default="weights/yolov5s.onnx",help='ONNX file path')
    parser.add_argument('--coco_anno', type=str, default="coco/instances_val2017.json", help="COCO annotation file")
    parser.add_argument('--image_size', type=int, default=20, help="COCO annotation file,total 5000")
    args = parser.parse_args()

    # If val2017 not exist, download and unzip
    if not os.path.exists(args.input_images_folder):
        print(f"Folder {args.input_images_folder} does not exist. Downloading val2017 dataset.")
        download_with_wget_and_unzip('http://images.cocodataset.org/zips/val2017.zip', './coco')
    
    # Load coco class names.
    classesFile = "coco/classes.txt"
    classes = None
    with open(classesFile, 'rt') as f:
        classes = f.read().rstrip('\n').split('\n')

    # Important, coco have 80 different classes, but original have 91, need mapping!
    COCO91_MAP = [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34,
            35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63,
            64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]

    # Load the model
    modelWeights = args.model_weights
    net = cv2.dnn.readNet(modelWeights)

    # Sort the image files to ensure consistent order
    image_files = os.listdir(args.input_images_folder)
    image_files.sort()
    # limit the number of images processed
    image_files = image_files[:args.image_size]
    all_detections = []
    image_id_list=[]
    # Empty list to store inference time
    inference_times = []
    # Create tdqm to visualise progress
    for image_file in tqdm(image_files):
        if image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(args.input_images_folder, image_file)
            input_image = cv2.imread(image_path)
            original_height, original_width = input_image.shape[:2]
            outputs = pre_process(input_image,  net)
            img,detections = post_process(input_image, outputs)
            # Put efficiency information.
            t, _ = net.getPerfProfile()
            inference_time = t * 1000.0 / cv2.getTickFrequency()
            label = 'Inference time: %.2f ms' % inference_time
            inference_times.append(inference_time)
            cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, RED, THICKNESS, cv2.LINE_AA)
            # Extract image id from the file name using regular expression
            match = re.search(r'(\d+)\.jpg$', image_file)
            image_id = int(match.group(1))
            image_id_list.append(image_id)
            output_path = os.path.join(args.output_images_folder, f"{image_id}.jpg")
            cv2.imwrite(output_path,img)
            # Convert detections to COCO format and append to the list
            coco_detections = generate_coco_format(detections, image_id)
            all_detections.extend(coco_detections)
    
    anno_json = args.coco_anno
    # For original annotations, loading, filtering and saving. Making ground truth number is consistent with processed image number
    with open(anno_json) as file:
        data = json.load(file)
    data['annotations'] = [anno for anno in data['annotations'] if anno['image_id'] in image_id_list]
    anno_json_filtered =args.coco_anno+'_filtered'
    with open(anno_json_filtered, 'w') as file:
        json.dump(data, file)
    
    pred_json = './coco/cocoval17_predictions.json'  # predictions json
    with open(pred_json, 'w') as f:
        json.dump(all_detections, f)
    anno = COCO(anno_json_filtered)  # init annotations api
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    # Calculate and print average inference time
    if inference_times:
        average_inference_time = sum(inference_times) / len(inference_times)
        print(f"Average Inference Time: {average_inference_time:.2f} ms")

