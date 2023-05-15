from yoloface import face_analysis
import cv2
import os

face = face_analysis()

def extractFaces():
    for dirname, _, filenames in os.walk('dataset'):
        for filename in filenames:

            img = cv2.imread(os.path.join(dirname, filename))

            if img is None:
                continue

            img, box, confidence = face.face_detection(image_path=os.path.join(dirname, filename), model='tiny')

            cnt = 0
            pos = 0
            for i, c in enumerate(confidence):
                if c > 0.95:
                    cnt += 1
                    pos = i

            if (cnt != 1) or len(box) < 1:
                continue

            x, y, h, w = [max(0, b) for b in box[pos]]
            
            face_img = img[y:y+h, x:x+w]
            
            path = os.path.join(dirname, filename).replace("dataset", "dataset_detected")

            newDir = dirname.replace("dataset", "dataset_detected")
            if not os.path.exists(newDir):
                os.makedirs(newDir)
            
            if os.path.exists(path):
                os.remove(path)

            cv2.imwrite(path, face_img)
