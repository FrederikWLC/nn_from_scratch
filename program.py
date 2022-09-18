from ml import Network, sigmoid, ReLu, np
import cv2
import string


class Program():
	def __init__(self):
		self.name = "Frederiks AI"
		self.cam = None
		self.ret, self.frame = None, None
		self.curr_itext = ""
		self.training_data = {} # {"label1":{[3,2,...], [3,5,...]...}, "label2":{[5,1,...], [5,5,...]...}, ...}
		self.h = 25
		self.w = 25
		self.net = Network(shape=[self.w*self.h,0],activations=[ReLu,sigmoid])
		self.prediction = ""
		self.predictions = []
		#self.img_counter = 0

	def run(self):
		self.cam = cv2.VideoCapture(0)
		cv2.namedWindow(self.name)
		while True:
			try:
				self.iter()
			except Break:
				break
		self.cam.release()
		cv2.destroyWindow(self.name)

	def iter(self):
		self.get_inputs()
		self.put_info()
		self.display_frame()

	def get_inputs(self):
		self.get_frame()
		self.get_predictions()
		self.check_keys()

	def get_frame(self):
		self.ret, self.frame = self.cam.read()
		if not self.ret:
			print("failed to grab frame")
			raise Break()
		h,w,c = self.frame.shape
		self.frame = cv2.resize(cv2.resize(self.frame, (25,25), interpolation = cv2.INTER_AREA),(w,h), interpolation = cv2.INTER_AREA)
		self.frame_bw = cv2.cvtColor(self.frame,cv2.COLOR_BGR2GRAY)

	def get_predictions(self):
		# Define input array from frame
		self.inp = [p for r in cv2.resize(self.frame_bw.reshape(-1), (25,25), interpolation = cv2.INTER_AREA) for p in r]
		# Get predictions with labels AND predicted raw output array from nn 
		if self.net.shape[-1]:
			self.predictions, self.output = self.net.predict_with_and_without_labels(self.inp)
			self.prediction = max(self.predictions.keys(), key=(lambda key: self.predictions[key])) if self.predictions else ""

	def check_keys(self):
		k = cv2.waitKey(1) 
		if chr(k%256) in string.ascii_lowercase + string.ascii_uppercase + " ":
			self.curr_itext += chr(k%256)
		if k%256 == 8:
			# BACKSPACE pressed
			self.curr_itext = self.curr_itext[:-1]
		if k%256 == 27:
			# ESC pressed
			print("Escape hit, closing...")
			raise Break()
		elif k%256 == 13 and self.curr_itext:
			if self.curr_itext in self.training_data.keys():
				self.training_data[self.curr_itext].append(self.inp)
			else:
				self.training_data[self.curr_itext] = [self.inp]
				self.net.add_neuron(len(self.net.shape)-1, label=self.curr_itext)
			print(f"Image under label '{self.curr_itext}' added to training data")
			#TRAIN
			self.net.train_from_labels(self.training_data)
			# Enter pressed
			"""
			img_name = "opencv_frame_{}.png".format(self.img_counter)
			cv2.imwrite(img_name, self.frame)
			print("{} written!".format(img_name))
			self.img_counter += 1
			"""
	def put_info(self):
		h,w = self.frame_bw.shape
		#cv2.rectangle(frame, (x,x), (x + w, y + h), (0,0,0), -1)
		cv2.putText(self.frame, text=f"INPUT: {self.curr_itext}",org=(0+10,int(h-10)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1, color=(0,0,255), thickness=3)
		if self.prediction:
			cv2.putText(self.frame, text=f"{self.prediction}",org=(int(0+10),int(h/10+10)), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=2, color=(0,255,0), thickness=5)
		
		lower_y = int(h/10+50)
		upper_y = int(h-25)
		diff_y = upper_y - lower_y
		num_of_labels = min(len(self.predictions),15)
		for i, label in enumerate(list(self.predictions)[:15]):
			x = int(10)
			y = int(lower_y + diff_y/num_of_labels*i)
			if i==14:
				cv2.putText(self.frame, text="...",org=(x,y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1-0.01*num_of_labels, color=(255,0,0), thickness=int(3-0.1*num_of_labels))
			else:
				cv2.putText(self.frame, text=f"{len(self.training_data[label])}x {label} : {self.predictions[label]}",org=(x,y), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1-0.01*num_of_labels, color=(255,0,0), thickness=int(3-0.1*num_of_labels))

	def display_frame(self):
		cv2.imshow(self.name, self.frame)


class Break(Exception):
    pass

def floor(x,decimals):
	return int(x) + float(str(x-int(x))[:decimals+2])

