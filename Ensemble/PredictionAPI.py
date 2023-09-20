#!/usr/bin/env python
# coding: utf-8

from flask import Flask, jsonify, abort, make_response, Response
from flask import request
#from flask_cors import CORS
import json
from json import JSONEncoder
from glob import glob
import argparse

import pprint
import requests
import numpy as np
import pandas as pd
import sys
import os
import pickle
from time import time
import os

from EnsembleModel import get_ensemble_model


def predict_models(smiles, models, reg_tasks, cls_tasks):
    results = models.predict(smiles)[0].numpy().astype(np.float64)
    tasks = reg_tasks + cls_tasks

    res=[]
    for i in range(len(smiles)):
        d = {}
        for j,task in enumerate(tasks):
            if np.isnan(results[j][i]):
                d[task]=None
            else:
                d[task]=results[j][i]
        res.append(d)
    return res


def get_weights(smiles, models, reg_tasks, cls_tasks):
    results = models.get_weights(smiles)[0].numpy().astype(np.float64)
    tasks = reg_tasks + cls_tasks

    d = {}
    for i,p in enumerate(tasks):
        d[p] =[list(map(float,x)) for x in results[i]]

    return d


def load_weights(models, path):
    for i in range(len(models)):
        models[i].load_weights(path[i])


def get_latest_folder(path):
    folders = glob(path+"/*")
    folders = sorted(folders, key=os.path.getmtime)
    return folders[0]

def get_model_list(path):
    folders = os.listdir(path)
    sorted_folders = sorted(folders, reverse=True)

    models=[]
    for f in sorted_folders:
        if f.isdecimal():
            models.append(f)

    return models


def get_model_path(path, model_name=None, model_ver=None, model_types=None):
    if model_ver is None:
        latest_folder = get_latest_folder(path)
    else:
        latest_folder = path + "/" + model_ver

    model_name=["ggt_model_pooling", "ggt_model"]
    model_path = latest_folder + "/result"
    full_path = [model_path + "/" + x for x in model_name]
    info_path = latest_folder + "/data/info"
    data_path = latest_folder + "/data/data.csv"

    return full_path, info_path, data_path

def get_models(path, model_ver=None):
    model_path, info_path, data_path = get_model_path(path, model_ver=model_ver)

    with open(info_path, "rb") as pf:
        info = pickle.load(pf)
        reg_tasks=info[0]
        cls_tasks=info[1]
        d_model=info[2]
        dff = info[3]

    models = get_ensemble_model(model_path[1], data_path, reg_tasks=reg_tasks, class_tasks=cls_tasks)

    return models, reg_tasks, cls_tasks


api = Flask(__name__)
#CORS(api, allow_headers = 'Content-Type')

@api.route('/predict', methods=['GET', 'POST'])
def predict():
    global current_model, models, cls_tasks, reg_tasks

    if request.method == 'POST':
        data = request.get_json()
        smiles = [x["SMILES"] for x in data["smiles"]]
        model = "Not defined"

    elif request.method == 'GET':
        smiles_string =  request.args.get("smiles", "Not defined")
        smiles = smiles_string.split(",")

        model = request.args.get("model", "Not defined")
        if model == "":
            model = "Not defined"

    else:
        smiles = "no mol"
        return "no mol"

    #model version check
    if model != "Not defined":
        try:
            models, reg_tasks, cls_tasks = get_models(path, model_ver=model)
            current_model = model
            print("load model:" + current_model)
        except:
            print("No directory or No model")
    else:
        if current_model != os.path.basename(get_latest_folder(path)):
            try:
                models, reg_tasks, cls_tasks = get_models(path)
                current_model = os.path.basename(get_latest_folder(path))
                print("load new model:" + current_model)
            except:
                print("No directory or No model")
    ans = predict_models(smiles, models, reg_tasks, cls_tasks)
    return jsonify(ans)


@api.route('/weight', methods=['GET', 'POST'])
def weight():
    global current_model, models, cls_tasks, reg_tasks
    if request.method == 'POST':
        data = request.get_json()
        smiles = [x["SMILES"] for x in data["smiles"]]

    elif request.method == 'GET':
        smiles_string =  request.args.get("smiles", "Not defined")
        smiles = smiles_string.split(",")
    else:
        smiles = "no mol"
        return "no mol"

    d = get_weights(smiles, models, reg_tasks, cls_tasks)
    return jsonify(d)


@api.route('/params', methods=['GET'])
def params():
    global current_model, models, cls_tasks, reg_tasks

    d = {}
    d["all"] = reg_tasks + cls_tasks
    d["regression"] = reg_tasks
    d["classification"] = cls_tasks
    return jsonify(d)


@api.route('/latest', methods=['GET'])
def latest():
    return jsonify(os.path.basename(get_latest_folder(path)))


@api.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)


if __name__ == '__main__':
    root_path = os.path.dirname(os.path.dirname(__file__))
    config_path = os.path.join(root_path, "Settings", "config.json")
    with open(config_path, "rt") as fp:
        config = json.loads(fp.read())

    path = config["Ensemble_train"]["save_path"]
    os.environ["CUDA_VISIBLE_DEVICES"]=config["use_GPU_No"]["Ensemble"]

    current_model= os.path.basename(get_latest_folder(path))
    models, reg_tasks, cls_tasks = get_models(path)

    if config["ssl"]["key"]=="":
        api.run(host=config["ip"]["Ensemble"],
                port=config["port"]["Ensemble"])
    else:
        api.run(host=config["ip"]["Ensemble"],
                port=config["port"]["Ensemble"],
                ssl_context=(config["ssl"]["crt"], config["ssl"]["key"]))

