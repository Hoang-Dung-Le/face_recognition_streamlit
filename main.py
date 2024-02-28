from deepface import DeepFace

dfs = DeepFace.find(img_path = "truonggiang_test.png", db_path = "./faces/", model_name='VGG-Face')
print(dfs)
