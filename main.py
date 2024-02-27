from deepface import DeepFace
import os
# print("ok")
# dfs = DeepFace.find(img_path = "face3.jpg", db_path = "./faces/", model_name='VGG-Face')
# print(dfs)

def recognize_images(source_folder, face_db_folder):
    total_similarity = 0
    total_images = 0

    # Duyệt qua từng tập tin ảnh trong thư mục nguồn
    for root, dirs, files in os.walk(source_folder):
        for file_name in files:
            image_path = os.path.join(root, file_name)

            # Nếu là tập tin ảnh, tiến hành nhận diện
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Nhận diện khuôn mặt trong ảnh
                result = DeepFace.find(img_path=image_path, db_path=face_db_folder, model_name='VGG-Face')
                print(result)
                break
            break

if __name__ == "__main__":
    print("========= Bắt đầu ==========")
    face_db_folder = "/content/Face Data/Data_Test"
    source_folder = "/content/Face Data/Data_Test1"  # Thư mục chứa ảnh khuôn mặt để so sánh

    recognize_images(source_folder, face_db_folder)
