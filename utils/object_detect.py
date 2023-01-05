from xml.dom.minidom import TypeInfo
import detectron2
import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator
from detectron2.utils.visualizer import Visualizer
from detectron2.utils.visualizer import ColorMode
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import os
import json
import cv2
import numpy as np
import math
import argparse
from PIL import Image
import pdf2image

yaml_path = './detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml'
weight_path = './checkpoints/model_detector.pth'

def pdf_convert(file, output):
    images = pdf2image.convert_from_path(file, output_folder=output, fmt="jpg")
    return images

# function that splits a list into n chunks for multiprocessing
def chunk(file_list, n_chunks):
    
    # make chunks of files to be distributed across processes
    chunks = []
    chunk_size = math.ceil(float(len(file_list))/n_chunks)
    for i in range(0, n_chunks-1):
        chunks.append(file_list[i*chunk_size:(i+1)*chunk_size])
    chunks.append(file_list[(n_chunks-1)*chunk_size:])
    
    return chunks

# defines a function for cropping all of the predicted visual content
def crop(detected, processed, dataset):

    json_dir = os.path.join(detected, dataset)
    json_files = os.listdir(json_dir)
    save_dir = os.path.join(processed, dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for json_file in json_files:
        filepath = os.path.join(json_dir, json_file)
        # we load the JSON
        with open(filepath) as f:
            predictions = json.load(f)
          
        # load in boxes
        boxes = predictions['boxes']
        scores = predictions['scores']
        classes = predictions['pred_classes']
        # ocrs = predictions['ocr']
        
        # grab filepath of image
        jpg_filepath = predictions['filepath']

        # open image
        im = Image.open(jpg_filepath)
        
        # empty list for storing embeddings
        img_embeddings = []
        
        # empty list or storing filepaths of extracted visual content
        content_filepaths = []

        # iterate through boxes, crop, and send to embedding
        for i in range(0, len(boxes)):
            box = boxes[i]
            pred_class = classes[i]
            score = scores[i]
            # ocr = ocrs[i]
            # if it's a headline or the confidence score is less than 0.5, we skip the cropping
            # if there is no ocr results in the box, we skip the cropping
            if pred_class == 5 or score < 0.5:
                img_embeddings.append("")
                content_filepaths.append("")
                continue
                
            # crop image according to box (converted from normalized coordinates to image coordinates)
            cropped = im.crop((box[0]*im.width, box[1]*im.height, box[2]*im.width, box[3]*im.height)).convert('RGB')
            # save cropped image to output directory
            cropped_filepath = json_file.replace(".json", "_" + str(i).zfill(3) + "_" + str(pred_class) + "_" + str(int(math.floor(100*score))).zfill(2) + ".jpg")
            cropped_filepath = os.path.join(save_dir, cropped_filepath)
            cropped.save(cropped_filepath)
            content_filepaths.append(cropped_filepath)

        # add filepaths of extracted visual content to output
        predictions['visual_content_filepaths'] = content_filepaths
        
        # we save the updated JSON
        with open(filepath, 'w') as f:
            json.dump(predictions, f)

def generate_predictions(input, output, dataset, batch_size=128, visualize=False):

    # navigates to correct directory (process is spawned in /notebooks)

    # sets up model for process
    image_dir = os.path.join(input, dataset)
    save_dir = os.path.join(output, dataset)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    images = os.listdir(image_dir)
    setup_logger()
    cfg = get_cfg()

    cfg.merge_from_file(yaml_path)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = 0.5
    cfg.MODEL.WEIGHTS = weight_path
    if(torch.cuda.is_available() == False):
        cfg.MODEL.DEVICE = 'cpu'
    # sets prediction score threshold - this is commented out and defaults to 0.05 in Detectron2
    # if you would like to adjust the threshold, uncomment and set to the desired value in [0, 1)
    # cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

    # sets number of object classes to 7
    # ("Illustration/Photograph", "Photograph", "Comics/Cartoon", "Editorial Cartoon", "Map", "Headline", "Ad")
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 7

    cfg.freeze()
    predictor = DefaultPredictor(cfg)
    # see:  https://github.com/facebookresearch/detectron2/issues/282 
    # (must load weights this way if using model)
    # model.train(False) 
    # construct batches
    batches = chunk(images, math.ceil(len(images)/batch_size))
    # iterate through images
    for batch in batches:

        # sets up inputs by loading in all files in batch
        # stores image dimensions
        dimensions = []
        outputs = []
        # iterate through files in batch
        for file in batch:
            file = os.path.join(image_dir, file)
            # read in image
            image = cv2.imread(file)
            image = np.asarray(image)
            # store image dimensions
            height, width, _ = image.shape
            dimensions.append([width, height])
            outputs.append(predictor(image))

            if(visualize):
                visualizer = Visualizer(image)
                instances = outputs[-1]["instances"]
                vis_output = visualizer.draw_instance_predictions(predictions=instances)
                vis_output.save(save_dir + '/' + file.split('/')[-1])
        # performs inference
        # saves predictions
        
        predictions = {}
        if(not visualize):
            # iterate over images in batch and save predictions to JSON
            for i in range(0, len(batch)):
                # saves filepath in format of ChronAm file structure
                predictions["filepath"] = os.path.join(image_dir, batch[i])
                
                # saves predictions
                # we first normalize the bounding box coordinates
                boxes = outputs[i]["instances"].get_fields()["pred_boxes"].to("cpu").tensor.tolist()
                normalized_boxes = []
                width = dimensions[i][0]
                height = dimensions[i][1]

                for box in boxes:
                    normalized_box = (box[0]/float(width), box[1]/float(height), box[2]/float(width), box[3]/float(height))
                    normalized_boxes.append(normalized_box)

                # saves additional outputs of predictions
                predictions["boxes"] = normalized_boxes
                predictions["scores"] = outputs[i]["instances"].get_fields()["scores"].to("cpu").tolist()
                predictions["pred_classes"] = outputs[i]["instances"].get_fields()["pred_classes"].to("cpu").tolist()
                
                with open(save_dir + '/' + batch[i].replace('.jpg','.json'), "w") as fp:
                    json.dump(predictions, fp)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--vis', action = 'store_true',
                    help='visualize images or not')
    parser.add_argument('--output', type=str, default='../static/data/detected/',
                    help='output folder path')    
    parser.add_argument('--input', type=str, default='../static/data/original/',
                    help='input folder path')
    parser.add_argument('--dataset', type=str, default='20140421-ST',
                    help='dataset name')
    parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
    args = parser.parse_args()
    filepath = os.path.join(args.input, args.dataset)
    files = os.listdir(filepath)
    for file in files:
        if(file.split('.')[-1] == 'pdf'):
            pth = os.path.join(filepath, file)
            pdf_convert(pth, filepath)
            os.remove(pth)
    generate_predictions(args.input, args.output, args.dataset, batch_size=args.batch_size, visualize=True)