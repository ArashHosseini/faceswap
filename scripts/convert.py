<<<<<<< HEAD
import cv2
import time
import re
import os


=======
#!/usr/bin python3
""" The script to run the convert process of faceswap """

import re
import os
>>>>>>> 958493a64a317388301e8e8a71202b90e6af2f2e
from pathlib import Path

from tqdm import tqdm

from scripts.fsmedia import Alignments, Images, Faces, Utils
from scripts.extract import Extract
from lib.utils import BackgroundGenerator, get_folder, get_image_paths

from plugins.PluginLoader import PluginLoader

<<<<<<< HEAD
class ConvertImage(DirectoryProcessor):
    filename = ''
    _init_time = None
    def create_parser(self, subparser, command, description):
        self.parser = subparser.add_parser(
            command,
            help="Convert a source image to a new one with the face swapped.",
            description=description,
            epilog="Questions and feedback: \
            https://github.com/deepfakes/faceswap-playground"
        )

    def add_optional_arguments(self, parser):
        parser.add_argument('-m', '--model-dir',
                            action=FullPaths,
                            dest="model_dir",
                            default="models",
                            help="Model directory. A directory containing the trained model \
                    you wish to process. Defaults to 'models'")

        parser.add_argument('-t', '--trainer',
                            type=str,
                            choices=("Original", "LowMem", "GAN"), # case sensitive because this is used to load a plug-in.
                            default="Original",
                            help="Select the trainer that was used to create the model.")

        parser.add_argument('-s', '--swap-model',
                            action="store_true",
                            dest="swap_model",
                            default=False,
                            help="Swap the model. Instead of A -> B, swap B -> A.")

        parser.add_argument('-c', '--converter',
                            type=str,
                            choices=("Masked", "Adjust", "GAN"), # case sensitive because this is used to load a plugin.
                            default="Masked",
                            help="Converter to use.")

        parser.add_argument('-D', '--detector',
                            type=str,
                            choices=("hog", "cnn"), # case sensitive because this is used to load a plugin.
                            default="cnn",
                            help="Detector to use. 'cnn' detects much more angles but will be much more resource intensive and may fail on large files.")

        parser.add_argument('-fr', '--frame-ranges',
                            nargs="+",
                            type=str,
                            help="frame ranges to apply transfer to e.g. For frames 10 to 50 and 90 to 100 use --frame-ranges 10-50 90-100. \
                            Files must have the frame-number as the last number in the name!"
                            )

        parser.add_argument('-d', '--discard-frames',
                            action="store_true",
                            dest="discard_frames",
                            default=False,
                            help="When used with --frame-ranges discards frames that are not processed instead of writing them out unchanged."
                            )

        parser.add_argument('-f', '--filter',
                            type=str,
                            dest="filter",
                            default="filter.jpg",
                            help="Reference image for the person you want to process. Should be a front portrait"
                            )

        parser.add_argument('-b', '--blur-size',
                            type=int,
                            default=2,
                            help="Blur size. (Masked converter only)")


        parser.add_argument('-S', '--seamless',
                            action="store_true",
                            dest="seamless_clone",
                            default=True,
                            help="Seamless mode. (Masked converter only)")

        parser.add_argument('-M', '--mask-type',
                            type=str.lower, #lowercase this, because its just a string later on.
                            dest="mask_type",
                            choices=["rect", "facehull", "facehullandrect"],
                            default="facehullandrect",
                            help="Mask to use to replace faces. (Masked converter only)")

        parser.add_argument('-e', '--erosion-kernel-size',
                            dest="erosion_kernel_size",
                            type=int,
                            default=None,
                            help="Erosion kernel size. (Masked converter only)")

        parser.add_argument('-sm', '--smooth-mask',
                            action="store_true",
                            dest="smooth_mask",
                            default=True,
                            help="Smooth mask (Adjust converter only)")

        parser.add_argument('-aca', '--avg-color-adjust',
                            action="store_true",
                            dest="avg_color_adjust",
                            default=True,
                            help="Average color adjust. (Adjust converter only)")
        return parser

    def process(self):
        # Original & LowMem models go with Adjust or Masked converter
        # GAN converter & model must go together
        # Note: GAN prediction outputs a mask + an image, while other predicts only an image
        model_name = self.arguments.trainer
        conv_name = self.arguments.converter

        if conv_name.startswith("GAN"):
            assert model_name.startswith("GAN") is True, "GAN converter can only be used with GAN model!"
        else:
            assert model_name.startswith("GAN") is False, "GAN model can only be used with GAN converter!"
=======
class Convert(object):
    """ The convert process. """
    def __init__(self, arguments):
        self.args = arguments
        self.output_dir = get_folder(self.args.output_dir)
>>>>>>> 958493a64a317388301e8e8a71202b90e6af2f2e

        self.images = Images(self.args)
        self.faces = Faces(self.args)
        self.alignments = Alignments(self.args)

        self.opts = OptionalActions(self.args, self.images.input_images)

<<<<<<< HEAD

        batch = BackgroundGenerator(self.prepare_images(), 1)
=======
    def process(self):
        """ Original & LowMem models go with Adjust or Masked converter
            Note: GAN prediction outputs a mask + an image, while other predicts only an image """
        Utils.set_verbosity(self.args.verbose)
>>>>>>> 958493a64a317388301e8e8a71202b90e6af2f2e

        if not self.alignments.have_alignments_file:
            self.generate_alignments()

        self.faces.faces_detected = self.alignments.read_alignments()

        model = self.load_model()
        converter = self.load_converter(model)

        batch = BackgroundGenerator(self.prepare_images(), 1)

        self.input_len = len(self.read_directory())
        self.done = 0
        for item in batch.iterator():
            self.convert(converter, item)

        Utils.finalize(self.images.images_found,
                       self.faces.num_faces_detected,
                       self.faces.verify_output)

<<<<<<< HEAD
    def convert(self, converter, item):
        try:
            start = time.time() 
            (filename, image, faces) = item
=======
    def generate_alignments(self):
        """ Generate an alignments file if one does not already
        exist. Does not save extracted faces """
        print('Alignments file not found. Generating at default values...')
        extract = Extract(self.args)
        extract.export_face = False
        extract.process()
>>>>>>> 958493a64a317388301e8e8a71202b90e6af2f2e

    def load_model(self):
        """ Load the model requested for conversion """
        model_name = self.args.trainer
        model_dir = get_folder(self.args.model_dir)
        num_gpus = self.args.gpus

        model = PluginLoader.get_model(model_name)(model_dir, num_gpus)

        if not model.load(self.args.swap_model):
            print("Model Not Found! A valid model must be provided to continue!")
            exit(1)

<<<<<<< HEAD
            output_file = get_folder(self.output_dir) / Path(filename).name
            cv2.imwrite(str(output_file), image)
            tt = time.time() - start
            estimate = (self.input_len-self.done) * tt
            print ("takes {0}, time left {1}".format(tt, str(datetime.timedelta(seconds=int(estimate)))))
            self.done += 1
        except Exception as e:
            print('Failed to convert image: {}. Reason: {}'.format(filename, e))
=======
        return model

    def load_converter(self, model):
        """ Load the requested converter for conversion """
        args = self.args
        conv = args.converter

        converter = PluginLoader.get_converter(conv)(model.converter(False),
                                                     trainer=args.trainer,
                                                     blur_size=args.blur_size,
                                                     seamless_clone=args.seamless_clone,
                                                     sharpen_image=args.sharpen_image,
                                                     mask_type=args.mask_type,
                                                     erosion_kernel_size=args.erosion_kernel_size,
                                                     match_histogram=args.match_histogram,
                                                     smooth_mask=args.smooth_mask,
                                                     avg_color_adjust=args.avg_color_adjust)
        return converter
>>>>>>> 958493a64a317388301e8e8a71202b90e6af2f2e

    def prepare_images(self):
        """ Prepare the images for conversion """
        filename = ""
        for filename in tqdm(self.images.input_images):
            if not self.check_alignments(filename):
                continue
            image = Utils.cv2_read_write('read', filename)
            faces = self.faces.get_faces_alignments(filename, image)
            if not faces:
                continue

            yield filename, image, faces

    def check_alignments(self, filename):
        """ If we have no alignments for this image, skip it """
        have_alignments = self.faces.have_face(filename)
        if not have_alignments:
            tqdm.write("No alignment found for {}, skipping".format(os.path.basename(filename)))
        return have_alignments

    def convert(self, converter, item):
        """ Apply the conversion transferring faces onto frames """
        try:
            filename, image, faces = item
            skip = self.opts.check_skipframe(filename)

            if not skip:
                for idx, face in faces:
                    image = self.convert_one_face(converter, (filename, image, idx, face))
            if skip != "discard":
                filename = str(self.output_dir / Path(filename).name)
                Utils.cv2_read_write('write', filename, image)
        except Exception as err:
            print("Failed to convert image: {}. Reason: {}".format(filename, err))

    def convert_one_face(self, converter, imagevars):
        """ Perform the conversion on the given frame for a single face """
        filename, image, idx, face = imagevars

        if self.opts.check_skipface(filename, idx):
            return image

        image = self.images.rotate_image(image, face.r)
        # TODO: This switch between 64 and 128 is a hack for now.
        # We should have a separate cli option for size
        image = converter.patch_image(image,
                                      face,
                                      64 if "128" not in self.args.trainer else 128)
        image = self.images.rotate_image(image, face.r, reverse=True)
        return image

class OptionalActions(object):
    """ Process the optional actions for convert """

    def __init__(self, args, input_images):
        self.args = args
        self.input_images = input_images

        self.faces_to_swap = self.get_aligned_directory()

        self.frame_ranges = self.get_frame_ranges()
        self.imageidxre = re.compile(r"(\d+)(?!.*\d)")

    ### SKIP FACES ###
    def get_aligned_directory(self):
        """ Check for the existence of an aligned directory for identifying
            which faces in the target frames should be swapped """
        faces_to_swap = None
        input_aligned_dir = self.args.input_aligned_dir

        if input_aligned_dir is None:
            print("Aligned directory not specified. All faces listed in the alignments file "
                  "will be converted")
        elif not os.path.isdir(input_aligned_dir):
            print("Aligned directory not found. All faces listed in the alignments file "
                  "will be converted")
        else:
            faces_to_swap = [Path(path) for path in get_image_paths(input_aligned_dir)]
            if not faces_to_swap:
                print("Aligned directory is empty, no faces will be converted!")
            elif len(faces_to_swap) <= len(self.input_images) / 3:
                print("Aligned directory contains an amount of images much less than the input, \
                        are you sure this is the right directory?")
        return faces_to_swap

    ### SKIP FRAME RANGES ###
    def get_frame_ranges(self):
        """ split out the frame ranges and parse out 'min' and 'max' values """
        if not self.args.frame_ranges:
            return None

        minmax = {"min": 0, # never any frames less than 0
                  "max": float("inf")}
        rng = [tuple(map(lambda q: minmax[q] if q in minmax.keys() else int(q), v.split("-")))
               for v in self.args.frame_ranges]
        return rng

    def check_skipframe(self, filename):
        """ Check whether frame is to be skipped """
        if not self.frame_ranges:
            return None
        idx = int(self.imageidxre.findall(filename)[0])
        skipframe = not any(map(lambda b: b[0] <= idx <= b[1], self.frame_ranges))
        if skipframe and self.args.discard_frames:
            skipframe = "discard"
        return skipframe

    def check_skipface(self, filename, face_idx):
        """ Check whether face is to be skipped """
        if self.faces_to_swap is None:
            return False
        face_name = "{}_{}{}".format(Path(filename).stem, face_idx, Path(filename).suffix)
        face_file = Path(self.args.input_aligned_dir) / Path(face_name)
        skip_face = face_file not in self.faces_to_swap
        if skip_face:
            print("face {} for frame {} was deleted, skipping".format(
                face_idx, os.path.basename(filename)))
        return skip_face
