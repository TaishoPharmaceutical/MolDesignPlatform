import os
import json
root_path = os.path.dirname(os.path.dirname(__file__))
webgui_path = os.path.join(root_path,"WebGUI")

config_path = os.path.join(root_path,"Settings","config.json")
with open(config_path, "rt") as pf:
    config = json.load(pf)

def url_sum(host, port, ssl_key):
    if "http" in host:
        url = f"{host}:{port}"
    else:
        if ssl_key == "":
            url=f"http://{host}:{port}"
        else:
            url=f"https://{host}:{port}"
    return url

ssl_key=config["ssl"]["key"]
MAIN_URL = url_sum(config["ip"]["WebGUI"], config["port"]["WebGUI"], ssl_key)
PREDICTION_API_URL = url_sum(config["ip"]["Ensemble"], config["port"]["Ensemble"], ssl_key)
REACTION_API_URL = url_sum(config["ip"]["Reactor"], config["port"]["Reactor"], ssl_key)
VQVAE_API_URL = url_sum(config["ip"]["VQVAE"], config["port"]["VQVAE"], ssl_key)
FRAGMENTVAE_API_URL = url_sum(config["ip"]["FragmentAE"], config["port"]["FragmentAE"], ssl_key)
FILTER_API_URL = url_sum(config["ip"]["Filter"], config["port"]["Filter"], ssl_key) + "/filt"
CHEMDOODLE_API_URL = url_sum(config["ip"]["Chemdoodle"], config["port"]["Chemdoodle"], ssl_key)

PER_COL=20
MAX_TAB=4

PADDING_REM = "0.65rem"

PANEL_TABLE_STYLE = {
    "width": "30%",
    "padding": "2rem 1rem",
    "background-color": "#fff",
}

PANEL_STYLE = lambda color: {
    "border-color": "#999",
    "background": color,
    "padding": PADDING_REM,
    "width": "100px",
}

FUNCTION_BUTTONS = [
    ["Reaction", "react"],
    ["FragmentAE", "linker"],
    ["VQVAEChem", "opt1"],
]


FUNCTION_STYLES = {
    "outline": "True",
    "color": "primary",
    "size": "sm",
    "style": { "width": "120px" }
}


TABLE_MENU_STYLES = {
    "width": "200px",
    "height": "70px",
    "vertical-align": "top",
}

SELECTION_BORDER_STYLE={
    0:"solid",
    1:"groove"
}

condition_list = [
    "linker", # length in atoms
    "hba",    # hydrogen acceptors
    "hbd",    # hydrogen donors
    "ar",     # aromatic rings
    "ali",    # aliphatic rings
    "rb",     # rotable bonds
    "chiral"  # chiral centers
]

condition_labels = {
    "linker": "L",
    "hba": "HBA",
    "hbd": "HBD",
    "ar": "ArR",
    "ali": "AliR",
    "rb": "RB",
    "chiral": "C"
}


def scaffold_preset():
    d = {}

    d["linker_min"] = 3
    d["linker_max"] = 5

    d["hba_min"] = -1
    d["hba_max"] = -1

    d["hbd_min"] = -1
    d["hbd_max"] = -1

    d["ar_min"] = 0
    d["ar_max"] = 2

    d["ali_min"] = 0
    d["ali_max"] = 2

    d["rb_min"] = -1
    d["rb_max"] = -1

    d["chiral_min"] = -1
    d["chiral_max"] = -1

    return d


def frag_link_preset():
    d = {}

    d["linker_min"] = 3
    d["linker_max"] = 6

    d["hba_min"] = -1
    d["hba_max"] = -1

    d["hbd_min"] = -1
    d["hbd_max"] = -1

    d["ar_min"] = 0
    d["ar_max"] = 0

    d["ali_min"] = 0
    d["ali_max"] = 0

    d["rb_min"] = -1
    d["rb_max"] = -1

    d["chiral_min"] = 0
    d["chiral_max"] = 0

    return d

def growing_preset():
    d = {}

    d["linker_min"] = 2
    d["linker_max"] = 6

    d["hba_min"] = -1
    d["hba_max"] = -1

    d["hbd_min"] = -1
    d["hbd_max"] = -1

    d["ar_min"] = -1
    d["ar_max"] = -1

    d["ali_min"] = -1
    d["ali_max"] = -1

    d["rb_min"] = -1
    d["rb_max"] = -1

    d["chiral_min"] = -1
    d["chiral_max"] = -1

    return d

SCAFFOLD_PRESET=scaffold_preset()
FLAG_LINK_PRESET=frag_link_preset()
GROWING_PRESET=growing_preset()

STRUCTURE_FILTER=["---", "rough", "basic", "strict"]
