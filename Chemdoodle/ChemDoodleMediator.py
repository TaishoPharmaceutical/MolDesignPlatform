#!/usr/bin/env python
# coding: utf-8

from flask import Flask, jsonify, abort, make_response, Response, render_template
from flask import request

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import Draw

import pprint
import requests
import numpy as np
import pandas as pd
import os
import shutil
import socket
from glob import glob
from time import time

import json


root_path = os.path.dirname(os.path.dirname(__file__))
chemdoodle_path = os.path.join(root_path, "Chemdoodle")
def get_config():
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "Settings", "config.json")
    with open(config_path, "rt") as fp:
        config = json.loads(fp.read())

    ip = config["ip"]["Chemdoodle"]
    port=config["port"]["Chemdoodle"]

    return ip, port, config


api = Flask(__name__)

@api.route('/save', methods=['GET', 'POST'])
def save():

    if request.method == 'POST':
        data = request.get_json()
        molblock = data["data"]
        file = data["hash"]
    else:
        smiles = "no mol"
        return "fail"


    # delete old files
    del_files = glob(f"{chemdoodle_path}/templates/*")
    times = [os.path.getctime(x) for x in del_files]
    del_files = [x for i,x in enumerate(del_files) if time()-times[i] > 3600*24]
    for x in del_files:
        shutil.rmtree(x)

    file = os.path.basename(file)
    with open(f"{chemdoodle_path}/templates/{file}/{file}.txt", "wt") as pf:
        pf.write(molblock)

    return ""


@api.route('/make', methods=['GET', 'POST'])
def make():
    if request.method == 'POST':
        data = request.get_json()
        molblock = data["data"].replace("\n","\\n")
        file = data["hash"]
    else:
        smiles = request.args.get("smiles", "Not_defined")
        file = request.args.get("hash", "Not_defined")

        if smiles == "Not_defined":
            return "fail"

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return "fail"

        molblock = Chem.MolToMolBlock(mol).replace("\n","\\n")

    folder = f"{chemdoodle_path}/templates/{file}"
    with open(f"{chemdoodle_path}/index.html", "rt") as pf:
        content = pf.read()

    ###delete old folders
    del_folders = glob(f"{chemdoodle_path}/templates/*")
    times = [os.path.getctime(x) for x in del_folders]
    del_folders = [x for i,x in enumerate(del_folders) if time()-times[i] > 3600*24]
    for x in del_folders:
        shutil.rmtree(x)
    ###

    #ip = socket.gethostbyname(socket.gethostname())
    os.makedirs(folder, exist_ok=True)
    content = content.replace("{molblock}",'%s'%molblock)

    ip, port, config = get_config()
    if config["ssl"]["key"] == "":
        content = content.replace("{address}",f"http://{ip}:{port}/save")
    else:
        content = content.replace("{address}",f"https://{ip}:{port}/save")
    with open(f"{folder}/index.html", "wt") as pf:
        pf.write(content)


    return render_template(f"{file}/index.html")


@api.route('/file', methods=['GET', 'POST'])
def file():
    if request.method == 'POST':
        data = request.get_json()
        molblock = data["data"].replace("\n","\\n")
        file = data["hash"]
    else:
        smiles = request.args.get("smiles", "Not_defined")
        file = request.args.get("hash", "Not_defined")

    path = os.path.join(chemdoodle_path,"templates",file,f"{file}.txt")
    return path


if __name__ == '__main__':
    ip, port, config = get_config()

    if config["ssl"]["key"]=="":
        api.run(host=ip, port=port)
    else:
        api.run(host=ip, port=port,
                ssl_context=(config["ssl"]["crt"], config["ssl"]["key"]))

