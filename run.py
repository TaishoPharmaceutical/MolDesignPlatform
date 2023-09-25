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
    sp = subprocess.Popen([python, chemdoodle_api])

    # Run Ensemble Predictor Model
    ensemble_api = os.path.join(root_path, "Ensemble", "PredictionAPI.py")
    sp = subprocess.Popen([python, ensemble_api])

    # Run FragmentAE
    fragmentAE_api = os.path.join(root_path, "FragmentAE", "FragmentAPI.py")
    sp = subprocess.Popen([python, fragmentAE_api])

    # Run Reactor
    reactor_api = os.path.join(root_path, "Reactor", "ReactorAPI.py")
    sp = subprocess.Popen([python, reactor_api])

    # Run Filter
    tasf_api = os.path.join(root_path, "Filter","filter_api.py")
    sp = subprocess.Popen([python, tasf_api])

    # Run VQVAE
    vqvae_api = os.path.join(root_path, "VQVAE", "ChemGenAPI.py")
    sp = subprocess.Popen([python, vqvae_api])

    time.sleep(20)

    # Run WebGUI
    webgui_api = os.path.join(root_path, "WebGUI", "WebGUI_run.py")
    sp = subprocess.Popen([python, webgui_api])
