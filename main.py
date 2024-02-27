from deepface import DeepFace
print("ok")
dfs = DeepFace.find(img_path = "face3.jpg", db_path = "./faces/", model_name='VGG-Face')
print(dfs)