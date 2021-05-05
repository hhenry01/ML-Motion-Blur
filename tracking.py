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

def show_img(img, title=None):
  if torch.is_tensor(img):
    img = img.cpu().permute(1,2,0)
  plt.imshow(img)
  if title:
    plt.title(title, color='w')
  plt.axis('off')

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
    # if classes:
    #   plt.text(x, y, classes[label.item()], backgroundcolor='r', c='w')
    i+=1
  plt.show()

def get_FRCNN_model(num_classes):
  model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
  in_features = model.roi_heads.box_predictor.cls_score.in_features
  model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
  return model

def load_model(filepath, device=torch.device('cpu')):
  model = get_FRCNN_model(len(classes)+1)
  model.load_state_dict(torch.load(filepath, map_location=device))
  model.eval()
  return model

def detect(model, img, device, confidence=0.6, iou=0.4):
  with torch.no_grad():
    img_tensor = torchvision.transforms.ToTensor()(img)
    detections = model([img_tensor.to(device)])
    print("Boxes detected:")
    print(detections)
    keeps = torchvision.ops.nms(detections[0]["boxes"], detections[0]["scores"], iou) # Not entirely sure what the iou_threshold is
    detections_filtered = []
    labels = []
    for i in range(detections[0]["boxes"].shape[0]):
      if detections[0]["scores"][i] >= confidence and i in keeps:
        detections_filtered.append(detections[0]["boxes"][i])
        labels.append(detections[0]["labels"][i])
    return detections_filtered, labels

# Example/Test
if __name__ == '__main__':
  print("Running")
  if torch.cuda.is_available():
    device = torch.device('cuda:0')
    print("On GPU")
  else:
    device = torch.device('cpu')
    print("On CPU")
  fp = sys.argv[1]
  model = load_model(fp, device)
  if sys.argv[3] == 0:
    img = sys.argv[2]
    img = Image.open(img).convert('RGB')
    detections, labels = detect(model, img, device)
    print(detections)
    print(labels)
    show_boxes(img, detections, labels, classes)
  else:
    vid = sys.argv[2]
    metadata = ffmpeg.probe(vid)["streams"][0]
    fps = metadata["r_frame_rate"]
    nb_frames = metadata["nb_frames"]
    length = metadata["duration"]
    vid = cv2.VideoCapture(vid)
    frame = vid.get(cv2.CAP_PROP_POS_FRAMES)

    for i in range(nb_frames):
      ready, curr_frame = cap.read()
      '''
      continue here
      '''
    vid.release()