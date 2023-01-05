import os
import json
import easyocr
import torch
import cv2
# # The next two cells define functions for extracting the OCR within each predicted box.
# 
# 1. The first cell defines the function for returning the proper OCR for a specific page.
# 2. The second cell defines the function for iterating over the JSON files containing the predictions.

# tolerance around box for testing whether OCR falls within bounds
WIDTH_TOLERANCE = 0.500
HEIGHT_TOLERANCE = 0.500

# given a file path and a list of bounding boxes, this function traverses the associated XML
# and returns the OCR within each bounding box
def get_ocr_for_file(true_img_filepaths, reader):
    # creates empty nested list fo storing OCR in each box
    ocr = [ [] for i in range(len(true_img_filepaths)) ]
    # # we now iterate over each bounding box
    for i in range(0, len(true_img_filepaths)):
        img = true_img_filepaths[i]
        if(len(img) == 0):
            ocr[i] = ""
            continue
        ocr_res = reader.readtext(img, paragraph=True)

        # we then iterate over each text box
        content = []
        for text_box in ocr_res:
            content.append(text_box[1])
        ocr[i] = ",".join(content)

    return ocr


def get_ocr(detected, dataset):
    json_path = os.path.join(detected, dataset)
    json_info = os.listdir(json_path)
    use_gpu = True    
    if(torch.cuda.is_available() == False):
        use_gpu = False
    reader = easyocr.Reader(['en', 'ch_sim'], gpu=use_gpu)
    # we now iterate through all of the predictions JSON files
    for json_filepath in json_info:
        json_filepath = os.path.join(json_path, json_filepath)
        # loads the JSON
        with open(json_filepath) as f:
            predictions = json.load(f)
        
        # pulls off relevant data fields from the JSON
        boxes = predictions['boxes']
        scores = predictions['scores']
        classes = predictions['pred_classes']

        # sets the number of predicted bounding boxes
        n_pred = len(scores)

        # we now find the JPG files corresponding to this predictions JSON
        # jpg_filepath = predictions['filepath']
        cropped_files = predictions['visual_content_filepaths']
        # stores list of OCR
        ocr = []

        # we only try to retrieve the OCR if there is one or more predicted box
        if n_pred > 0:
            ocr = get_ocr_for_file(cropped_files, reader)

        # adds the ocr field to the JSON metadata for the page
        predictions['ocr'] = ocr

        # we save the updated JSON
        with open(json_filepath, 'w') as f:
            json.dump(predictions, f)