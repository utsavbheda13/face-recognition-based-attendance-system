FaceRecognitionBasedAttendanceSystem
-----------------------------------------
The face recognition-based attendance system is a biometric technology that uses facial features to verify and authenticate the identity of an individual. This technology has gained immense popularity due to its ability to accurately identify individuals, even in large groups, with high speed and convenience. In this project, we aim to develop a face recognition-based attendance system that will automate the attendance management process in educational institutions and workplaces.

MTCNN (Multi-task Cascaded Convolutional Neural Networks) and YOLO (You Only Look Once) are two popular face detection algorithms.MTCNN is a three-stage cascaded approach that first detects candidate regions, then refines them, and finally classifies them as faces or non-faces. YOLO is a single-stage approach that directly predicts bounding boxes and confidence scores for faces in an image. MTCNN is a more accurate face detection algorithm than YOLO, but it is also slower. YOLO is a faster face detection algorithm than MTCNN, but it is less accurate. So, for training, we are using YOLO as it contains only one image for recognition, and for the actual attendance marking mechanism we are using MTCNN to detect multiple faces.

VGGFace and ResNet are two popular convolutional neural network (CNN) architectures that have been used for face recognition. VGGFace is a relatively simple architecture that consists of a stack of convolutional layers and max pooling layers. The VGGFace architecture has been shown to be effective for face recognition. ResNet is a more complex architecture than VGGFace, and it consists of a stack of residual blocks. Residual blocks are a way of making CNNs deeper without encountering the vanishing gradient problem. ResNet has been shown to be more accurate than VGG for face recognition, and also VGGFace has more parameters than ResNet. So for the training we have preferred Resnet.

Steps to execute the code:-
-----------------------------
Command to install python libraries **"pip install -r requirements.txt"**

Command to execute project **"python3 ui.py"**
