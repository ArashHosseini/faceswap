try:
	from lib.faces_detect import detect_faces
except RuntimeError as e:
	raise Exception("possible out of memory!!!")

from plugins.PluginLoader import PluginLoader
import cv2
import os
import argparse

def extract(input_path,  output_path):
	files = os.listdir(input_path)
	if not len(files): raise Exception("no files inside {0}!!!".format(input_path))
	extractor = PluginLoader.get_extractor("Align")()
	for n, _file in enumerate(files):
		_file_id = os.path.join(input_path, _file)
		#print (_file_id)
		output = os.path.join(output_path, _file)#"{0}.{1}".format(n, _file.split(".")[1]))
		if not os.path.exists(output):
			if os.path.isfile(_file_id):
				print ("file {0}/{1}".format(n,len(files)), _file_id)
				image = cv2.imread(_file_id)
				try:
					for (idx, face) in enumerate(detect_faces(image)):
						resized_image = extractor.extract(image, face, 256)
						cv2.imwrite(output, resized_image)
				except Exception as e:
					print('Failed to extract from image: {}. Reason: {}'.format(_file, e))
		else:
			print("Jump ", output)

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='extract faces')
	parser.add_argument('-i', '--input',
						action="store",
						dest='input',
						help='set the input directory path to source images',
						default="")
	parser.add_argument('-o', '--output',
						action="store",
						dest='output',
						help='set the output directory path',
						default="")
	_arg = parser.parse_args()
	if not os.path.exists(_arg.input): raise Exception("input directory {0} not exists!!!".format(_arg.input))
	if not os.path.exists(_arg.output): raise Exception("output directory {0} not exists!!!".format(_arg.output))
	extract(_arg.input, _arg.output)
	

