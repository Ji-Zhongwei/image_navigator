from utils import *
import argparse
import os
from img2vec_pytorch import Img2Vec

def generate_embeddings(detected, dataset, gpu_id=-1):
    
    # load in img2vec
    # we choose resnet embeddings
    if gpu_id == -1:
        img2vec_resnet_50 = Img2Vec(cuda=False, model='resnet-50') 
        img2vec_resnet_18 = Img2Vec(cuda=False, model='resnet-18') 
    else:
        img2vec_resnet_50 = Img2Vec(cuda=True, model='resnet-50') 
        img2vec_resnet_18 = Img2Vec(cuda=True, model='resnet-18') 
    json_filepaths = os.path.join(detected, dataset)
    # iterate through the JSON files
    for json_filepath in json_filepaths:
        
        # we load the JSON
        with open(json_filepath) as f:
            predictions = json.load(f)

        # load in boxes
        boxes = predictions['boxes']
        scores = predictions['scores']
        classes = predictions['pred_classes']
        cropped_filepaths = predictions['visual_content_filepaths']

        # grab filepath of image
        jpg_filepath = predictions['filepath']

        # empty list for storing embeddings
        resnet_50_embeddings = []
        resnet_18_embeddings = []

        # iterate through boxes, crop, and send to embedding
        for i in range(0, len(boxes)):

            box = boxes[i]
            pred_class = classes[i]
            score = scores[i]
            
            # if it's a headline or confidence score is less than 0.5, we skip the embedding generation
            if pred_class == 5 or score < 0.5:
                resnet_50_embeddings.append([])
                resnet_18_embeddings.append([])
                continue

            cropped_filepath = cropped_filepaths[i]
            # reformat to use flat file directory
            cropped_filepath = cropped_filepath.replace("/", "_")
            
            # open cropped image
            im = Image.open(cropped_filepath).convert('RGB')
            # generate embedding using img2vec
            embedding_resnet_50 = img2vec_resnet_50.get_vec(im, tensor=False)
            embedding_resnet_18 = img2vec_resnet_18.get_vec(im, tensor=False)
            # add to list (render embedding numpy array as list to enable JSON serialization)
            resnet_50_embeddings.append(embedding_resnet_50.tolist())
            resnet_18_embeddings.append(embedding_resnet_18.tolist())
            
        embeddings_json = {}
        embeddings_json['filepath'] = predictions['filepath']
        embeddings_json['visual_content_filepaths'] = predictions['visual_content_filepaths']
        # add embeddings to output
        embeddings_json['resnet_50_embeddings'] = resnet_50_embeddings
        embeddings_json['resnet_18_embeddings'] = resnet_18_embeddings

        # we save the updated JSON
        with open(json_filepath[:-5] + "/embeddings.json", 'w') as f:
            json.dump(embeddings_json, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--processed', type=str, default='./static/data/processed/',
                    help='detected folder path')   
    parser.add_argument('--detected', type=str, default='./static/data/detected/',
                    help='detected folder path')    
    parser.add_argument('--input', type=str, default='./static/data/original/',
                    help='input folder path')
    parser.add_argument('--dataset', type=str, default='20140421-ST',
                    help='dataset name')
    parser.add_argument('--batch-size', type=int, default=128,
                    help='batch size')
    args = parser.parse_args()

    filepath = os.path.join(args.input, args.dataset)
    files = os.listdir(filepath)
    # convert pdf to images
    for file in files:
        if(file.split('.')[-1] == 'pdf'):
            print('converting pdf file: {} into images...'.format(file))
            pth = os.path.join(filepath, file)
            pdf_convert(pth, filepath)
            os.remove(pth)
    # detect objections in images
    generate_predictions(args.input, args.detected, args.dataset, batch_size=args.batch_size)
    crop(args.detected, args.processed, args.dataset)
    get_ocr(args.detected, args.dataset)
    # generate_embeddings(args.detected, args.dataset)
    print('preprocess done!')