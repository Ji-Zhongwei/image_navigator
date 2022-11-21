# Flask
from flask import Flask, Blueprint, request, jsonify, flash, render_template, redirect, send_from_directory, make_response, url_for
from werkzeug.datastructures import ImmutableMultiDict
import flask

# scikit-learn
from sklearn.metrics.pairwise import cosine_distances
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.svm import LinearSVC

# NLTK
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# NumPy
import numpy as np

# various Flask libraries
from flask_paginate import Pagination, get_page_args
from forms import DataSearchForm

# vanilla Python imports
from collections import Counter, OrderedDict
from functools import reduce
import datetime
import zipfile
import glob
import json
import math
import time
import csv
import sys
import io
import os
import re

N_PER_PAGE = 100

N_PREDICTIONS = 200

# dataset
data = "./static/data/detected/"
# embedding = "./static/data/detected/20140421-ST/embeddings/"

app = Flask(__name__)

# load data in the beginning
def load_data():
    global image_data
    # global embedding_data
    image_data = []
    # embedding_data = []
    json_files = []
    datasets = os.listdir(data)
    for dataset in datasets:
        datapath = os.path.join(data, dataset)
        for file in os.listdir(datapath):
            filepath = os.path.join(datapath, file)
            json_files.append(filepath)
    
    for json_file in json_files:
        with open(json_file) as f:
            predictions = json.load(f)
        for i in range(len(predictions["visual_content_filepaths"])):
            
            image_data.append({"uuid": i,
                                "filepath": predictions["visual_content_filepaths"][i],
                                "ocr": predictions["ocr"][i]})
    
    # json_files = os.listdir(embedding)
    # for json_file in embedding:
    #     filepath = os.path.join(embedding, json_file)
    #     with open(filepath) as f:
    #         embeddings = json.load(f)
    #     embedding_data.append(embeddings)
    for i in range(0, len(image_data)):
        lowered = image_data[i]['ocr'].lower()
        image_data[i]['lowered_ocr'] = lowered

# search homepage
@app.route('/', methods=['GET', 'POST'])
def search():
    search = DataSearchForm(request.args)
    if request.method == 'GET':
        if search.data['search'] is None:
            return render_template('index.html', form=search)
        return search_results(search)
        
    return render_template('index.html', form=search)

# search
def perform_search(metadata, search):
    res = []
    for md in metadata:
        # if the search query isn't empty, we find OCR containing query
        if len(search) > 0:
            if 'lowered_ocr' in md.keys():
                # OCR is lowered upon load b/c the lowering operation is quite costly
                if not search in md['lowered_ocr']:
                    continue

        # if the image has made it this far, it's a valid search result
        res.append(md)
    return res

# search results page
@app.route('/results', methods=['GET'])
def search_results(search_params):

    page, per_page, offset = get_page_args(page_parameter='page',
                                           per_page_parameter='per_page')

    if request.method == 'GET':

        # parse query string
        search = request.args["search"]
        if search is None:
            search = ""
        else:
            search = search.lower()

        metadata = image_data
        res = perform_search(metadata, search)
        search_suggestions = []
        # IF USING GENSIM #
        ## generates search suggestions using word2vec (genism)
        # if search != "":
        #     try:
        #         suggestions = word2vec_model.most_similar(search.lower().split())
        #         for suggestion in suggestions:
        #             search_suggestions.append(suggestion[0])
        #             # can also do stemming here!
        #     except:
        #         pass
        ###################

        pagination = Pagination(page=page, per_page=N_PER_PAGE, total=len(res))
        return render_template('search.html',
                                results=res[(page-1)*N_PER_PAGE:page*N_PER_PAGE],
                                form=search_params,
                                page=page,
                                per_page=N_PER_PAGE,
                                pagination=pagination,
                                search_suggestions=search_suggestions
                                )


if __name__ == '__main__':
    load_data()
    app.run(port=8060)
