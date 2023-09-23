#!/usr/bin/env python
# coding: utf-8


import requests
import shutil
import os
import subprocess
import json
import pathlib
import time
import argparse

root_path = os.getcwd()
config_path = os.path.join(root_path,"Settings","config.json")
with open(config_path, "rt") as fp:
    config = json.load(fp)


# ### Download Chemdoodle
#
# ##### 1) visit chemdoodle site and download Chemdoodle js. https://web.chemdoodle.com/
# ##### 2) Unzip download file and change the name of the unzipped-folder to ChemDoodle.
# ##### 3) Put the folder in Chemdoodle/static

## This code is Python code to execute the processing in the previous sentence.
## Before running this code, visit the Chemdoodle download site.
def download_chemdoodle():
    url="https://web.chemdoodle.com/downloads/ChemDoodleWeb-9.5.0.zip"
    folder = "Chemdoodle/static"
    filename=f"{folder}/ChemDoodleWeb-9.5.0.zip"
    urlData = requests.get(url).content

    with open(filename ,mode='wb') as fp:
        fp.write(urlData)

    shutil.unpack_archive(filename, folder)
    os.remove(filename)


# ### Make Reactants list
def make_reactants_list():
    python = config["python"]
    prepro = os.path.join(root_path, "Reactor/make_reactants_table.py")

    sp = subprocess.Popen([
        python,
        prepro,
        "--df", config["Reactor_data"],
    ])
    sp.wait()


# ### Train Ensemble Predictor
def train_Ensemble():
    python = config["python"]
    train_path = os.path.join(root_path, "Ensemble/train_ensemble.py")

    sp = subprocess.Popen([
        python,
        train_path,
        "--df", config["Ensemble_train"]["data_path"],
        "--save", os.path.join(config["Ensemble_train"]["save_path"],str(time.strftime("%Y%m%d"))),
        "--gpu", config["Ensemble_train"]["use_gpu"],
        "--zscore", config["Ensemble_train"]["z_score"],
        "--batch", config["Ensemble_train"]["batch_size"],
        "--epoch", config["Ensemble_train"]["epoch"],
    ])
    sp.wait()


# ### Train FragmentAE
def train_FragmentAE():
    python = config["python"]
    train_path = os.path.join(root_path, "FragmentAE/train_FragmentAE.py")

    sp = subprocess.Popen([
        python,
        train_path,
        "--df", config["FragmentAE_train"]["data_path"],
        "--save", config["FragmentAE_train"]["save_path"],
        "--gpu", config["FragmentAE_train"]["use_gpu"],
        "--batch", config["FragmentAE_train"]["batch_size"],
        "--epoch", config["FragmentAE_train"]["epoch"],
    ])
    sp.wait()


# ### Train VQVAE
def train_VQVAE():
    python = config["python"]
    train_path = os.path.join(root_path, "VQVAE/train_VQVAE.py")

    sp = subprocess.Popen([
        python,
        train_path,
        "--df", config["VQVAE_train"]["data_path"],
        "--save", config["VQVAE_train"]["save_path"],
        "--gpu", config["VQVAE_train"]["use_gpu"],
        "--batch", config["VQVAE_train"]["batch_size"],
        "--VQTrainEpoch", config["VQVAE_train"]["VQ_train_epoch"],
        "--VAETrainEpoch", config["VQVAE_train"]["VAE_train_epoch"],
    ])
    sp.wait()


runtype="""
download_chemdoodle: Download ChemDoodleJS,
make_reactants_list: Make the reactants list from the reagents csv,
train_ensemble: Training of Ensemble predictor model,
train_fragment: Training of FragmentAE model,
train_VQVAE: Training of VQVAE model,
All: run all above
"""


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--RunType", help=runtype, type=str, default="All")
    args = parser.parse_args()

    if args.RunType=="All" or args.RunType=="download_chemdoodle":
        download_chemdoodle()

    if args.RunType=="All" or args.RunType=="make_reactants_list":
        make_reactants_list()

    if args.RunType=="All" or args.RunType=="train_ensemble":
        train_Ensemble()

    if args.RunType=="All" or args.RunType=="train_fragment":
        train_FragmentAE()

    if args.RunType=="All" or args.RunType=="train_VQVAE":
        train_VQVAE()
