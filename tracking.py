'''
This file contains the code that performs object detection and tracking.
When running the file, the arguments are:

1: path the the Faster RCNN Model
2: file/path name, either to an image or video
3: Enter 0 for image, 1 for video

Ex.
python3 tracking.py Model.pt test_samples/person_standing.jpg 0
'''

import sys
import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cv2
import ffmpeg
import parser

TEST = True
TEST_PLAYBACK = False


'''
Only preps the image for showing. Need to include
"plt.show()" after all extra processing is done.
'''
def show_img(img, title=None):
  if torch.is_tensor(img):
    img = img.cpu().permute(1,2,0)
  plt.imshow(img)
  if title:
    plt.title(title, color='w')
  plt.axis('off')

'''
Displays bounding boxes on an image.
'''
def show_boxes(img, boxes, labels=None, classes=None):
  show_img(img)
  ax = plt.gca()
  # xmin, xmax = ax.get_xlim()
  # ymax, ymin = ax.get_ylim()
  # isx, isy = xmax-xmin, ymax-ymin
  i = 0
  for box in boxes:
    x = box[0]
    y = box[1]
    w = box[2] - x
    h = box[3] - y
    bbox = patches.Rectangle((x, y), w, h, ec='r', fc='none')
    ax.add_patch(bbox)
    if labels:
      if classes:
        name = classes[labels[i].item()]
      else:
        name = labels[i].item()
      plt.text(x, y, name, backgroundcolor='r', c='w')
    i+=1
  plt.show()

'''
Acquires a Faster RCNN model.
'''
def get_FRCNN_model(num_classes):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model

'''
Load a Faster RCNN with custom pretrained weights.
'''
def load_model(filepath, num_classes, device=torch.device('cpu')):
  model = get_FRCNN_model(num_classes)
  model.load_state_dict(torch.load(filepath, map_location=device))
  model.eval()
  return model

'''
With a given Pytorch Faster RCNN model and image, identifies bounding 
boxes for the model's relevant classes. Results are filtered based
on the confidence/score the model gives a detection as 
well as its iou_threshold. *Not too sure what iou_threshold is.

Returns a list of bounding box coordinate tensors and a list of 
predicted labels for each box.
'''
def detect(model, img, device, confidence=0.6, iou=0.4):
  with torch.no_grad():
    img_tensor = torchvision.transforms.ToTensor()(img)
    detections = model([img_tensor.to(device)])
    if TEST:
      print("Boxes detected:")
      print(detections)
    keeps = torchvision.ops.nms(detections[0]["boxes"], detections[0]["scores"], iou) 
    detections_filtered = []
    labels = []
    for i in range(detections[0]["boxes"].shape[0]):
      if detections[0]["scores"][i] >= confidence and i in keeps:
        detections_filtered.append(detections[0]["boxes"][i])
        labels.append(detections[0]["labels"][i])
    return detections_filtered, labels

'''
Given a Pytorch Faster RCNN model and video path, identifies
boudning boxes for the model's relevant classes throughout each
frame of the video. Same filters as the detect() function.

Returns a list of all detections and a list of all labels done by
the model in each frame.
'''
def track(model, vid_path, device, confidence=0.6, iou=0.4): 
  if TEST:
    metadata = ffmpeg.probe(vid_path)["streams"][0]
    fps = eval(parser.expr(metadata["r_frame_rate"]).compile())
    # nb_frames = int(metadata["nb_frames"])
    # length = float(metadata["duration"])
    # print(str(fps) + " " + str(nb_frames) + " " +str(length))
    frametime = int((1 / fps) * 1000) # Frametime is 1 / fps. Multiply by 1000 to get it in ms.
    print(frametime)

  cap = cv2.VideoCapture(vid_path)
  frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
  detections = []
  labels = []
  while True:
    ready, curr_frame = cap.read()
    while not ready:
      print("Frame not ready")
      cap.set(cv2.CAP_PROP_POS_FRAMES, frame-1)
      ready, curr_frame = cap.read()

    img = Image.fromarray(curr_frame)
    
    if TEST_PLAYBACK and TEST:
      cv2.imshow('video', curr_frame)
      cv2.waitKey(frametime)
      # Only have the above or below commented out at a time
      # show_img(img)
      # plt.show(block=False)
      # plt.pause(frametime)
      # plt.close()
    
    curr_detections, curr_labels = detect(model, curr_frame, device, confidence, iou)
    detections.append(curr_detections)
    labels.append(curr_labels)

    if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT): # Break when video is done
      break

  cap.release()
  assert len(detections) == len(labels), "Mismatch between number of detections and labels."
  return detections, labels

# Example/Test
if __name__ == '__main__':
  # Background is class 0 and is predefined in Pytorch
  classes = {
    1: 'shin',
    2: 'thigh',
    3: 'torso',
    4: 'forearm',
    5: 'uparm',
    6: 'head',
    7: 'ball',
    8: 'cup',
    9: 'juice',
    10: 'rubikscube',
    11: 'hand',
    12: 'bird',
    13: 'motorcycle',
    14: 'bike',
    15: 'car',
    16: 'toy-dog'
}

  print("Running")
  if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("On GPU")
  else:
    device = torch.device('cpu')
    print("On CPU")
  fp = sys.argv[1]
  model = load_model(fp, len(classes)+1, device)

  if sys.argv[3] == '0':
    print("Working on image")
    img = sys.argv[2]
    img = Image.open(img).convert('RGB')
    detections, labels = detect(model, img, device)
    if TEST:
      print(detections)
      print(labels)
    show_boxes(img, detections, labels, classes)
  else:
    print("Working on video")
    vid_path = sys.argv[2]
    # On my CPU it took ~35 minutes to process a ~17 second 720p video.
    detections, labels = track(model, vid_path, device)
    if TEST:
      print(detections)

    