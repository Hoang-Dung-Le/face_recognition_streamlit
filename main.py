from deepface import DeepFace

dfs = DeepFace.find(img_path = "truonggiang_test.png", db_path = "./faces/")
print(dfs)