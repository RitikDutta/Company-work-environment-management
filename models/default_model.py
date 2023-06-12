import os

class Default_Model():
	def __init__(self):
		pass
	def default_model(self):
		os.popen('cp models/default_faces_embeddings.npz models/faces_embeddings.npz')
		os.popen('cp models/default_face_SVC_model.pkl models/face_SVC_model.pkl')