from deepface import DeepFace

dfs = DeepFace.find(img_path = "truonggiang_test.jpg", db_path = "./faces/", model_name='VGG-Face')
print(dfs)

# def get_image_paths(directory):
#     image_paths = []
#     valid_image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Các định dạng ảnh hỗ trợ

#     for root, dirs, files in os.walk(directory):
#         for file in files:
#             file_path = os.path.join(root, file)
#             _, file_extension = os.path.splitext(file_path)
#             if file_extension.lower() in valid_image_extensions:
#                 image_paths.append(file_path)
#     return image_paths

# if __name__ == "__main__":
#     directory_path = '/content/Face Data/Data_Test1'
#     image_paths = get_image_paths(directory_path)

#     # In ra đường dẫn của tất cả các ảnh trong thư mục
#     for path in image_paths:
#         print(path)
#         dfs = DeepFace.find(img_path = path, db_path = "/content/Face Data/Data_Test", model_name='VGG-Face')
#         print(dfs)
#         break