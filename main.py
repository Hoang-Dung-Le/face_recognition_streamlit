from deepface import DeepFace
import os
# print("ok")
# dfs = DeepFace.find(img_path = "face3.jpg", db_path = "./faces/", model_name='VGG-Face')
# print(dfs)
def recognize_images(source_folder, face_db_folder):
    # Duyệt qua từng tập tin ảnh trong thư mục nguồn
    for root, dirs, files in os.walk(source_folder):
        for file_name in files:
            image_path = os.path.join(root, file_name)

            # Nếu là tập tin ảnh, tiến hành nhận diện
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
                # Nhận diện khuôn mặt trong ảnh
                result = DeepFace.find(img_path=image_path, db_path=face_db_folder, model_name='VGG-Face')

                # In thông tin nhận diện và độ chính xác
                print(f"\nNhận diện cho ảnh {file_name}:")
                print("Kết quả:", result)

                # Nếu có ít nhất một kết quả
                if result:
                    # In ra độ chính xác của kết quả đầu tiên
                    print("Độ chính xác:", result[0]['Similarity'])

if __name__ == "__main__":
    source_folder = "/content/Face Data/Face Dataset"
    face_db_folder = "/content/Face Data/Data_Test" # Thư mục chứa ảnh khuôn mặt để so sánh

    recognize_images(source_folder, face_db_folder)