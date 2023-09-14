#!/usr/bin/env python
# coding: utf-8

import subprocess
import json
import time
import os


if __name__=="__main__":

    # Run Chemdoodle
    root_path = os.getcwd()
    config_path = os.path.join(root_path,"Settings","config.json")
    with open(config_path, "rt") as fp:
        config = json.load(fp)

    python = config["python"]
    chemdoodle_api = os.path.join(root_path, "Chemdoodle","ChemDoodleMediator.py")
    cmd = " ".join([python, chemdoodle_api])
    sp = subprocess.Popen(cmd)


    # Run Ensemble Predictor Model
    ensemble_api = os.path.join(root_path, "Ensemble", "PredictionAPI.py")
    cmd = " ".join([python, ensemble_api])
    sp = subprocess.Popen(cmd)


    # Run FragmentAE
    fragmentAE_api = os.path.join(root_path, "FragmentAE", "FragmentAPI.py")
    cmd = " ".join([python, fragmentAE_api])
    sp = subprocess.Popen(cmd)


    # Run Reactor
    reactor_api = os.path.join(root_path, "Reactor", "ReactorAPI.py")
    cmd = " ".join([python, reactor_api])
    sp = subprocess.Popen(cmd)


    # Run Filter
    tasf_api = os.path.join(root_path, "Filter","filter_api.py")
    cmd = " ".join([python, tasf_api])
    sp = subprocess.Popen(cmd)


    # Run VQVAE
    vqvae_api = os.path.join(root_path, "VQVAE", "ChemGenAPI.py")
    cmd = " ".join([python, vqvae_api])
    sp = subprocess.Popen(cmd)


    # Run WebGUI
    time.sleep(20)

    webgui_api = os.path.join(root_path, "WebGUI", "WebGUI_run.py")
    cmd = " ".join([python, webgui_api])
    sp = subprocess.Popen(cmd)

