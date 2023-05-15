from mtcnn import MTCNN
from yoloface import face_analysis
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
tensorflow.keras.utils.disable_interactive_logging()

# change this
mapping = {0: 'Aamir_Khan', 1: 'Abhay_Deol', 2: 'Abhishek_Bachchan', 3: 'Aftab_Shivdasani', 4: 'Aishwarya_Rai', 5: 'Ajay_Devgn', 6: 'Akshay_Kumar', 7: 'Akshaye_Khanna', 8: 'Alia_Bhatt', 9: 'Ameesha_Patel', 10: 'Amitabh_Bachchan', 11: 'Amrita_Rao', 12: 'Amy_Jackson', 13: 'Anil_Kapoor', 14: 'Anushka_Sharma', 15: 'Anushka_Shetty', 16: 'Arjun_Kapoor', 17: 'Arjun_Rampal', 18: 'Arshad_Warsi', 19: 'Asin', 20: 'Ayushmann_Khurrana', 21: 'Bhumi_Pednekar', 22: 'Bipasha_Basu', 23: 'Bobby_Deol', 24: 'Deepika_Padukone', 25: 'Disha_Patani', 26: 'Emraan_Hashmi', 27: 'Esha_Gupta', 28: 'Farhan_Akhtar', 29: 'Govinda', 30: 'Hrithik_Roshan', 31: 'Huma_Qureshi', 32: 'Ileana_D╬У├З├ЦCruz', 33: 'Irrfan_Khan', 34: 'Jacqueline_Fernandez', 35: 'John_Abraham', 36: 'Juhi_Chawla', 37: 'Kajal_Aggarwal', 38: 'Kajol', 39: 'Kangana_Ranaut', 40: 'Kareena_Kapoor', 41: 'Karisma_Kapoor', 42: 'Kartik_Aaryan', 43: 'Katrina_Kaif', 44: 'Kiara_Advani', 45: 'Kriti_Kharbanda', 46: 'Kriti_Sanon', 47: 'Kunal_Khemu', 48: 'Lara_Dutta', 49: 'Madhuri_Dixit', 50: 'Manoj_Bajpayee', 51: 'Mrunal_Thakur', 52: 'Nana_Patekar', 53: 'Nargis_Fakhri', 54: 'Naseeruddin_Shah', 55: 'Nushrat_Bharucha', 56: 'Paresh_Rawal', 57: 'Parineeti_Chopra', 58: 'Pooja_Hegde', 59: 'Prabhas', 60: 'Prachi_Desai', 61: 'Preity_Zinta', 62: 'Priyanka_Chopra', 63: 'R_Madhavan', 64: 'Rajkummar_Rao', 65: 'Ranbir_Kapoor', 66: 'Randeep_Hooda', 67: 'Rani_Mukerji', 68: 'Ranveer_Singh', 69: 'Richa_Chadda', 70: 'Riteish_Deshmukh', 71: 'Saif_Ali_Khan', 72: 'Salman_Khan', 73: 'Sanjay_Dutt', 74: 'Sara_Ali_Khan', 75: 'Shah_Rukh_Khan', 76: 'Shahid_Kapoor', 77: 'Shilpa_Shetty', 78: 'Shraddha_Kapoor', 79: 'Shreyas_Talpade', 80: 'Shruti_Haasan', 81: 'Sidharth_Malhotra', 82: 'Sonakshi_Sinha', 83: 'Sonam_Kapoor', 84: 'Suniel_Shetty', 85: 'Sunny_Deol', 86: 'Sushant_Singh_Rajput', 87: 'Taapsee_Pannu', 88: 'Tabu', 89: 'Tamannaah_Bhatia', 90: 'Tiger_Shroff', 91: 'Tusshar_Kapoor', 92: 'Uday_Chopra', 93: 'Vaani_Kapoor', 94: 'Varun_Dhawan', 95: 'Vicky_Kaushal', 96: 'Vidya_Balan', 97: 'Vivek_Oberoi', 98: 'Yami_Gautam', 99: 'Zareen_Khan'}
face = face_analysis()

num_classes = 100
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

num_classes = 100
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
  mtcnn = MTCNN()
  global face
  try:
    img = cv2.imread(image_file)
    if img is not None:
      return face.face_detection(image_path=image_file, model='tiny')

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = mtcnn.detect_faces(img_rgb)
    return img, faces
  except cv2.error as e:
    print("Image file:", image_file)
    print("opencv error:", e)
    exit(-1)

def save_detected_image(img, face, number, dir):
  """Save detected image to temporary directory and send it for recognition"""
  x, y, height, width = face
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
  """Recognition of faces from detected faces"""
  num_classes = 100                                           # change this
  faces = load_image_and_preprocess(detected_dir)
  global model
  output = model(faces)
  _, predicted = torch.max(output.data, 1)
  return predicted.tolist()

def recognition(input_image):
  img, faces, confidences = detect_faces(input_image)       # check working by giving corrupted image
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
    if confidences[index] >= 0.95:
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
