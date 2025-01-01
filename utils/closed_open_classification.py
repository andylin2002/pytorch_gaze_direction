from PIL import Image
import numpy as np
import dlib
import os
import shutil

# 調整開眼比例ㄉ門檻
eyes_threshold = 0.12

# 初始化 Dlib 的人臉檢測器和 68 點預測器
face_detector = dlib.get_frontal_face_detector()
MODEL_PATH = os.path.join("utils/eyes_catch_model")
MODEL_FILE_PATH = os.path.join(MODEL_PATH, "shape_predictor_68_face_landmarks.dat")
predictor = dlib.shape_predictor(MODEL_FILE_PATH)

# 定義分類的資料夾路徑
pictures_dir = "./client_pictures"
base_dir = "./log/closed_open_classification"
open_dir = os.path.join(base_dir, "open")
closed_dir = os.path.join(base_dir, "closed")
faces_dir = os.path.join(base_dir, "closed_faces")  # 儲存人臉的資料夾

# 創建分類資料夾（如果不存在）
os.makedirs(open_dir, exist_ok=True)
os.makedirs(closed_dir, exist_ok=True)
os.makedirs(faces_dir, exist_ok=True)

# 檢測眼睛是否睜開的函數
def is_eye_open(landmarks, left_eye_indices, right_eye_indices):
    def calculate_eye_aspect_ratio(eye_points):
        # EAR 計算公式：((p2 - p6) + (p3 - p5)) / (2.0 * (p1 - p4))
        A = ((eye_points[1].y - eye_points[5].y) ** 2 + (eye_points[1].x - eye_points[5].x) ** 2) ** 0.5
        B = ((eye_points[2].y - eye_points[4].y) ** 2 + (eye_points[2].x - eye_points[4].x) ** 2) ** 0.5
        C = ((eye_points[0].y - eye_points[3].y) ** 2 + (eye_points[0].x - eye_points[3].x) ** 2) ** 0.5
        return (A + B) / (2.0 * C)

    left_eye = [landmarks.part(idx) for idx in left_eye_indices]
    right_eye = [landmarks.part(idx) for idx in right_eye_indices]

    left_ear = calculate_eye_aspect_ratio(left_eye)
    right_ear = calculate_eye_aspect_ratio(right_eye)
    print(f"* 第 {idx + 1} 個人臉 -> 左眼睜眼比例: {left_ear:.3f}, 右眼睜眼比例: {right_ear:.3f}")

    # EAR 閾值（0.18 通常適合判斷閉眼）
    return left_ear > eyes_threshold and right_ear > eyes_threshold

# 遍歷目錄中的所有圖片
for file_name in os.listdir(pictures_dir):
    if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
        file_path = os.path.join(pictures_dir, file_name)

        # 使用 PIL 讀取圖像並轉換為 RGB 格式
        image = Image.open(file_path).convert("RGB")
        img = np.array(image)

        # 確保圖像類型為 uint8
        if img.dtype != np.uint8:
            img = (img / img.max() * 255).astype(np.uint8)

        # Dlib 人臉偵測
        faces = face_detector(img, 0)
        print(f"{file_name}：偵測到 {len(faces)} 張臉")
        
        all_open = True

        # 檢查每張臉是否睜眼
        for idx, face in enumerate(faces):
            # 保存偵測到的人臉圖像
            

            # 傳遞 NumPy 圖像和矩形區域到 predictor
            landmarks = predictor(img, face)
            left_eye_indices = [36, 37, 38, 39, 40, 41]
            right_eye_indices = [42, 43, 44, 45, 46, 47]

            eye_open = is_eye_open(landmarks, left_eye_indices, right_eye_indices)

            if not eye_open:
                face_image = image.crop((face.left(), face.top(), face.right(), face.bottom()))
                face_file_name = f"{os.path.splitext(file_name)[0]}_face_{idx + 1}.jpg"
                face_file_path = os.path.join(faces_dir, face_file_name)
                face_image.save(face_file_path)
                all_open = False
                break
        
        if len(faces)==0:
            all_open = False

        # 根據結果分類圖片
        if all_open:
            shutil.copy2(file_path, os.path.join(open_dir, file_name))
            print(f"圖片 {file_name} 已分類到 'open' 資料夾")
        else:
            shutil.move(file_path, os.path.join(closed_dir, file_name))
            print(f"圖片 {file_name} 已分類到 'closed' 資料夾")
    print("=============")
print("分類完成，並且閉眼的人臉已儲存！")