import sys
import torch, torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

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

def show_boxes(img, boxes):
  show_img(img)
  ax = plt.gca()
  # xmin, xmax = ax.get_xlim()
  # ymax, ymin = ax.get_ylim()
  # isx, isy = xmax-xmin, ymax-ymin
  for box in boxes:
    x = box[0]
    y = box[1]
    w = box[2] - x
    h = box[3] - y
    bbox = patches.Rectangle((x, y), w, h, ec='r', fc='none')
    ax.add_patch(bbox)
    # if classes:
    #   plt.text(x, y, classes[label.item()], backgroundcolor='r', c='w')
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
    detections_filtered =[]
    for i in range(detections[0]["boxes"].shape[0]):
      if detections[0]["scores"][i] >= confidence and i in keeps:
        detections_filtered.append(detections[0]["boxes"][i])
    return detections_filtered

# Example/Test
print("Running")
if torch.cuda.is_available():
  device = torch.device('cuda:0')
  print("On GPU")
else:
  device = torch.device('cpu')
  print("On CPU")
fp = sys.argv[1]
model = load_model(fp, device)
img = sys.argv[2]
img = Image.open(img).convert('RGB')
detections = detect(model, img, device)
print(detections)
show_boxes(img, detections)