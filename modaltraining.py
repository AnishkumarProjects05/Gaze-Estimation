import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import mediapipe as mp

# -----------------------------
# 1. Load Images + Labels
# -----------------------------

BASE_DIR = "MPIIFaceGaze_preprocessed"
label_file = os.path.join(BASE_DIR, "Label", "p00.label")

image_data = []
gaze_labels = []

with open(label_file, "r") as f:
    lines = f.readlines()

for line in lines[1:]:
    parts = line.strip().split()

    face_img_path = parts[0].replace("\\", os.sep)
    full_img_path = os.path.join(BASE_DIR, "Image", face_img_path)

    img = cv2.imread(full_img_path)

    if img is None:
        print("Image not found:", full_img_path)
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = img.astype(np.float32) / 255.0

    gaze_values = parts[-2].split(",")

    gaze_x = float(gaze_values[-2])
    gaze_y = float(gaze_values[-1])

    image_data.append(img)
    gaze_labels.append([gaze_x, gaze_y])

image_data = np.array(image_data, dtype=np.float32)
gaze_labels = np.array(gaze_labels, dtype=np.float32)

print("Images loaded:", image_data.shape)
print("Labels loaded:", gaze_labels.shape)

# -----------------------------
# 2. Train-Test Split
# -----------------------------

train_images, test_images, train_labels, test_labels = train_test_split(
    image_data,
    gaze_labels,
    test_size=0.2,
    random_state=42
)

print("Images split for training and testing")

# -----------------------------
# 3. Dataset Class
# -----------------------------

class FaceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        img = torch.tensor(img, dtype=torch.float32)
        img = img.permute(2, 0, 1)

        label = self.labels[idx]

        return img, label


train_dataset = FaceDataset(train_images, train_labels)
test_dataset = FaceDataset(test_images, test_labels)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

sample_batch, _ = next(iter(train_loader))
print("Sample batch shape:", sample_batch.shape)

# -----------------------------
# 4. CNN Model
# -----------------------------

class GazeCNN(nn.Module):
    def __init__(self):
        super(GazeCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        self.fc1 = nn.Linear(64 * 56 * 56, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


model = GazeCNN()
print("GazeCNN model created successfully")

# -----------------------------
# 5. Training
# -----------------------------

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    model.train()
    running_loss = 0.0

    for inputs, targets in train_loader:
        outputs = model(inputs)

        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Loss: {avg_loss:.4f}")

print("Training completed successfully")

# -----------------------------
# 6. Save Model
# -----------------------------

torch.save(model.state_dict(), "gaze_model.pth")
print("Model saved as gaze_model.pth")

# -----------------------------
# 7. Webcam Iris Tracking
# -----------------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not opened")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    h, w, _ = frame.shape

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)

    eye_detected = False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            right_iris_points = []
            left_iris_points = []

            for idx in [469, 470, 471, 472]:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                right_iris_points.append((x, y))

            for idx in [474, 475, 476, 477]:
                landmark = face_landmarks.landmark[idx]
                x = int(landmark.x * w)
                y = int(landmark.y * h)
                left_iris_points.append((x, y))

            if len(right_iris_points) == 4 and len(left_iris_points) == 4:
                

                right_center = np.mean(right_iris_points, axis=0).astype(int)
                left_center = np.mean(left_iris_points, axis=0).astype(int)
                # Right eye openness
                right_top = face_landmarks.landmark[159]
                right_bottom = face_landmarks.landmark[145]

                # Left eye openness
                left_top = face_landmarks.landmark[386]
                left_bottom = face_landmarks.landmark[374]

                right_eye_open = abs(right_top.y - right_bottom.y)
                left_eye_open = abs(left_top.y - left_bottom.y)

                threshold = 0.008

                cv2.circle(frame, tuple(right_center), 5, (0, 255, 0), -1)
                cv2.circle(frame, tuple(left_center), 5, (0, 255, 0), -1)

                cv2.putText(
                    frame,
                    f"Right Eye: x={right_center[0]}, y={right_center[1]}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Left Eye: x={left_center[0]}, y={left_center[1]}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                print(
                    f"Right Eye: x={right_center[0]}, y={right_center[1]} | "
                    f"Left Eye: x={left_center[0]}, y={left_center[1]}"
                )
                if right_eye_open > threshold and left_eye_open > threshold:
                    eye_detected = True
                else:
                    eye_detected = False

    if not eye_detected:
        cv2.putText(
            frame,
            "No eye detected",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

    cv2.imshow("Retina / Iris Tracker", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()