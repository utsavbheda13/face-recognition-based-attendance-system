from mtcnn import MTCNN
from PIL import Image
from facenet_pytorch import InceptionResnetV1
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import torch
import os
import pandas as pd
import datetime
import tensorflow
import torch.nn as nn
import pickle
tensorflow.keras.utils.disable_interactive_logging()

# change this
with open('index_to_name.pickle', 'rb') as f:
  mapping = pickle.load(f)
with open('n_classes.pickle', 'rb') as f:
  num_classes = pickle.load(f)
num_classes = num_classes['num_classes']
mtcnn = MTCNN()

class Flatten(nn.Module):
  def __init__(self):
    super(Flatten, self).__init__()
      
  def forward(self, x):
    x = x.view(x.size(0), -1)
    return x
        
class normalize(nn.Module):
  def __init__(self):
    super(normalize, self).__init__()
      
  def forward(self, x):
    x = F.normalize(x, p=2, dim=1)
    return x

model = InceptionResnetV1(num_classes=num_classes, classify=True)
model = nn.Sequential(*list(model.children())[:-5])
model.avgpool_1a = nn.AdaptiveAvgPool2d(output_size=1)
model.last_linear = nn.Sequential(
  Flatten(),
  nn.Linear(in_features=1792, out_features=128, bias=False),
  normalize()
)
model.logits = nn.Linear(128, num_classes)
model.softmax = nn.Softmax(dim=1)
model_path = "resnet_model.pt"
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()

def detect_faces(image_file):
  """ Detect all images from input image file and return image and face boxes"""
  global mtcnn
  try:
    img = cv2.imread(image_file)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mtcnn.detect_faces(img_rgb)
    return img, faces
  except cv2.error as e:
    print("Image file:", image_file)
    print("opencv error:", e)
    exit(-1)

def save_detected_image(img, face, number, dir):
  """Save detected image to temporary directory and send it for recognition"""
  x, y, width, height = face['box']
  face_img = img[y:y+height, x:x+width]
  save_path = os.path.join(dir, f"{number}.jpg")
  cv2.imwrite(save_path, face_img)
  return save_path

def preprocess_and_transform(image):
  """Resize image and normalize along all direction"""
  transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
  ])
  return transform(image)

def load_image_and_preprocess(detected_dir):
  """Read detected images and apply preprocess and convert it to tensor"""
  images = []
  for filename in os.listdir(detected_dir):
    img_path = os.path.join(detected_dir, filename)
    if os.path.isfile(img_path):
      image = Image.open(img_path)
      image = preprocess_and_transform(image)
      images.append(image)
  return torch.stack(images)

def predict(detected_dir):
  """Recognition faces from detected faces"""
  faces = load_image_and_preprocess(detected_dir)
  global model
  output = model(faces)
  _, predicted = torch.max(output.data, 1)
  return predicted.tolist()

def recognition(input_image):
  img, faces = detect_faces(input_image)       # check working by giving corrupted image
  detected_dir = "detected"

  if not os.path.exists(detected_dir):
    os.mkdir(detected_dir)

  files = os.listdir(detected_dir)

  # Loop through each file and remove it
  for file_name in files:
      file_path = os.path.join(detected_dir, file_name)
      try:
          if os.path.isfile(file_path):
              os.remove(file_path)
      except Exception as e:
          print(f"Error removing file {file_path}: {e}")


  for index, face in enumerate(faces):
    if face['confidence'] >= 0.95:
      face_path = save_detected_image(img, face, index, detected_dir)
  predicted = predict(detected_dir)
  result = []
  for pred in predicted:
    result.append({
      "Name": mapping[pred],
      "Roll": pred})
  print(result)
  return result


def markAttendanceInSheet(result):
  df = pd.read_csv('sheet.csv')

  today = datetime.date.today().strftime('%Y-%m-%d')
  # Check if the date column already exists
  if today not in df.columns:
      # If it doesn't exist, create a new column with today's date
      df[today] = ''

  # Check if each student is present in the list, and mark them as present if so
  for stud in result:
      # Check if the student is present in the list of present students
      if stud['Name'] in df['student'].values:
          # If the student is present, mark them as present on 2023-05-22
          df.loc[df['student'] == stud['Name'], today] = 'present'


  # Save the updated DataFrame to a new CSV file
  df.to_csv('sheet.csv', index=False)
