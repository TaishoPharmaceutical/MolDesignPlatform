import os
import ast
import json
from flask import jsonify
import requests
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import Descriptors

from werkzeug.utils import secure_filename

import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, callback_context, no_update, Input, Output, State, ALL, MATCH
from dash.exceptions import PreventUpdate

from app import app
from utils import get_pred_df, get_weight, get_url_smiles, get_pred_panel, get_column, get_downloader, url_smiles_to, get_file_path
from utils import PER_COL, MAX_TAB
import hashlib
import urllib

import chem_util
import mol2dtoolkit
import chem_selector

import socket
from time import time, sleep
from panel import getHovers, get_pred_panel_list, getImgPanel, getFuncPanel
from constants import SCAFFOLD_PRESET, FLAG_LINK_PRESET, GROWING_PRESET, MAIN_URL, PREDICTION_API_URL, REACTION_API_URL, FRAGMENTVAE_API_URL, CHEMDOODLE_API_URL, VQVAE_API_URL, root_path, webgui_path
from panel_modal import REACTION_TYPE

MAX_USER_DATA=10000
REACTION_TYPE_CONDENSATION = 0
REACTION_TYPE_SUZUKI_COUPLING = 1
REACTION_TYPE_SnAr = 2

DATA_CLEAR_INTERVAL=3600 * 2 #sec


OPTION_LABEL={
    0:"Main",
    1:"Reaction",
    2:"FragmentVAE",
    3:"Peripherals"
}

param_dict = {
    "solv": "Solubility(ug/mL)",
    "Cyto10": "Cyto_10uM",
    "Cyto50": "Cyto_50uM",
    "hERG1": "hERG_1uM",
    "hERG10": "hERG_10uM",
}

PANEL_TABLE_STYLE = {
    "width": "30%",
    "padding": "2rem 1rem",
    "background-color": "#fff",
}

global_progress_value = None
global_progress_flag = False


user_data=[[] for i in range(MAX_USER_DATA)]
user_att_data=[[] for i in range(MAX_USER_DATA)]
time_stamps=[0 for i in range(MAX_USER_DATA)]
def get_user_id():
    cur = time()
    for i in range(len(time_stamps)):
        if cur - time_stamps[i] > DATA_CLEAR_INTERVAL:
            user_data[i]=[]
            user_att_data[i]=[]

    time_diff = [cur - x for x in time_stamps]
    user_id = np.argmax(time_diff)

    return user_id
#smiles_df = None

@app.callback(
    Output("react_modal", "is_open"),
    Output("react-button", "children"),
    Output({"type":"result", "index":1}, "children"),

    Input({"type": "panel_react_func", "index": ALL}, "n_clicks"),

    Input("MolWt_min", "value"),
    Input("MolWt_max", "value"),

    Input("ArNum_min", "value"),
    Input("ArNum_max", "value"),

    Input("react-radio", "value"),
    Input("react-button", "children"),
    State("react_modal", "is_open")
)
def react_button_func(n_clicks,
                      MolWt_min, MolWt_max,
                      ArNum_min, ArNum_max,
                      react_type, react_component, is_open):
    #print(MolWt_min, MolWt_max)

    # 機能ボタンを押されたときはlen() == 1となる。
    # それ以外の場合、len()はテーブルに表示している化合物数になる。
    # 機能ボタンを押したとき以外はモーダルを表示しないためのif
    if len(callback_context.triggered)!=1:
        return False, react_component, no_update
    elif len(callback_context.triggered)==1 and callback_context.triggered[0]["value"] == 0:
        return False, react_component, no_update
    elif not callback_context.triggered:
        return False, react_component, no_update
    elif callback_context.triggered[0]["prop_id"].split(".")[1] == "value":
        return True, react_component, no_update

    trigger_dict = ast.literal_eval(callback_context.triggered[0]["prop_id"].split(".")[0])

    if trigger_dict["index"] == "close":
        return False, no_update, no_update

    smi = trigger_dict["index"].split("_")[1]

    react_button_form = html.Div([
        dbc.Button("Submit", id={'type': 'panel_react_func', 'index': f"submit_{smi}"}, n_clicks=0),
        "   ",
        dbc.Button("Close", id={'type': 'panel_react_func', 'index': "close"}, n_clicks=0),
    ])

    if ("submit" in trigger_dict["index"]):
        smi = trigger_dict["index"].split("_")[1]

        MolWt_min = MolWt_min if not None else -1
        MolWt_max = MolWt_max if not None else -1
        ArNum_min = ArNum_min if not None else -1
        ArNum_max = ArNum_max if not None else -1
        print(react_type)
        r = requests.post(f"{REACTION_API_URL}/react", proxies={"http": None}, verify=False,
                 json={
                     "smiles": smi,
                     "reaction": react_type,
                     "MolWt_min": MolWt_min,
                     "MolWt_max": MolWt_max,
                     "ArNum_min": ArNum_min,
                     "ArNum_max": ArNum_max,
                 })

        if r.text == "":
            return True, react_button_form, r.text

        return False, react_button_form, json.dumps(r.json()) #show_pred_panel()へ続く

    return True, react_button_form, no_update

@app.callback(
    #Output({"type": "linker-selector", "index": "test"}, 'figure'),
    Output("linker_result", 'value'),
    Input({"type": "linker-selector", "index": ALL}, 'selectedData'),
)
def callback(sel):
    temp = callback_context.triggered[0]["value"]

    if temp is None:
        return ""
    elif temp == [None]:
        return ""

    trigger_dict = ast.literal_eval(callback_context.triggered[0]["prop_id"].split(".")[0])
    smi = trigger_dict["index"]

    res = [s["pointNumber"] for s in sel[0]["points"]]

    core_smi, subst_smi = mol2dtoolkit.get_parts(smi, res)
    subst_smi = subst_smi+".."+core_smi
    return subst_smi

    res_fig = get_figure(self.df, self.df.index, sel, self.bond_list)

    if sel == None:
        return ret_style, res_fig, None

    res = [s["pointNumber"] for s in sel["points"]]

    #frag = set(range(len(mol.GetAtoms()))) - set(res)

    core_smi, subst_smi = mol2dtoolkit.get_parts(self.smi, res)
    subst_smi = subst_smi+".."+core_smi

    if subst_smi.count(".") > 1:
        return ret_style, res_fig, "Error: too many fragments"

    return ret_style, res_fig, subst_smi


# linker modal の開閉
@app.callback(
    Output("linker_modal", "is_open"),
    Output("linker-sel2", "children"),
    Input({"type": "panel_linker_func", "index": ALL}, "n_clicks"),
    State("linker_modal", "is_open"),
    prevent_initial_call=True
)
def linker_modal_open_close(n_clicks, is_open):
    if len(callback_context.triggered)!=1:
        return no_update, no_update
    elif len(callback_context.triggered)==1 and callback_context.triggered[0]["value"] == 0:
        return no_update, no_update

    if is_open:
        return False, no_update

    trigger_dict = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trigger_dict == "":
        return no_update, no_update
    trigger_dict = ast.literal_eval(trigger_dict)

    key = trigger_dict["index"].split("_")
    if len(key) == 2:
        smi = key[1]
    else:
        return no_update, no_update

    selector = dcc.Graph(
        figure=chem_selector.make_chem_selector(smi),
        id={'type': 'linker-selector', 'index': smi},
        config={'displayModeBar': False},
        style={"border": "1px solid", "width": "300px", "height": "300px"}
    )
    return True, selector

def get_fragmentvae_result(select_smi):
    frags = select_smi.split("..")
    if frags[-1] == "invalid":
        return no_update

    r = requests.post(f"{FRAGMENTVAE_API_URL}/generate", proxies={"http": None},
             data={"smi1": frags[1], "smi2": frags[0]}, verify=False)
    smiles = r.json()

    return smiles


linker_functions={
    "hopping":"scaffold",
    "linking":"frag",
    "growing":"growing"
}

# linkingの実行
@app.callback(
    Output({"type":"result", "index":2}, "children"),
    Input({"type": "panel_linker_func", "index": ALL}, "n_clicks"),
    State("choose_ai_model", "value"),
    State("linker_result", "value"),
    State({'type': 'linker-selector', 'index': ALL}, "id"),
    State({"type":"filter", "index":"linking"}, "value"),
    prevent_initial_call=True
)
def linker_button_func(n_clicks, ai_model, select_smi, idx, filter_level):
    if not callback_context.triggered:
        return no_update

    if "invalid" in select_smi:
        return no_update

    trigger_dict = ast.literal_eval(callback_context.triggered[0]["prop_id"].split(".n_clicks")[0])
    func = trigger_dict["index"]

    if func in linker_functions.keys():
        smiles = get_fragmentvae_result(select_smi)
        if filter_level != "---":
            smiles = chem_util.get_filtered_smiles(smiles, filter_level)
        return ",".join(smiles)

    return no_update


@app.callback(
    Output("opt1_modal", "is_open"),
    Output("opt-button", "children"),
    Output({"type":"result", "index":3}, "children"),
    Input({"type": "panel_opt1_func", "index": ALL}, "n_clicks"),
    Input("opt-button", "children"),
    State("opt1_modal", "is_open"),
    State({"type":"filter", "index":"opt"}, "value")
)
def opt1_button_func(n_clicks, opt_component, is_open, filter_level):

    # 機能ボタンを押されたときはlen() == 1となる。
    # それ以外の場合、len()はテーブルに表示している化合物数になる。
    # 機能ボタンを押したとき以外はモーダルを表示しないためのif
    if len(callback_context.triggered)!=1:
        return False, opt_component, ""
    elif len(callback_context.triggered)==1 and callback_context.triggered[0]["value"] == 0:
        return False, opt_component, ""

    if not callback_context.triggered:
        return False, opt_component, no_update
    elif callback_context.triggered[0]["prop_id"].split(".")[1] == "value":
        return True, opt_component, no_update

    trigger_dict = ast.literal_eval(callback_context.triggered[0]["prop_id"].split(".")[0])

    if trigger_dict["index"] == "close":
        return False, no_update, no_update

    smi = trigger_dict["index"].split("_")[1]

    opt_button_form = html.Div([
        dbc.Button("Submit", id={'type': 'panel_opt1_func', 'index': f"submit_{smi}"}, n_clicks=0),
        "   ",
        dbc.Button("Close", id={'type': 'panel_opt1_func', 'index': "close"}, n_clicks=0),
    ])

    if ("submit" in trigger_dict["index"]):
        smi = trigger_dict["index"].split("_")[1]

        r = requests.post(f"{VQVAE_API_URL}/vae", proxies={"http": None},
                 json={"smiles": smi}, verify=False)

        if filter_level != "---":
            smiles = chem_util.get_filtered_smiles(r.json(), filter_level)
        else:
            smiles = r.json()

        res = ",".join(smiles)

        return False, opt_button_form, res

    return True, opt_button_form, no_update


@app.callback(
    Output({"type":"result", "index":0}, "children"),
    Output("smiles", "value"),
    Output("cashe_smiles", "value"),
    Input("submit", "n_clicks"),
    Input("url", "search"),
    State("smiles", "value"),
    State("url", "pathname"),
    State("cashe_smiles", "value"),
    prevent_inidital_call = True
)
def push_sub_button(n_clicks, search, value, pathname, cashe_smiles):
    smis= ""
    if pathname == "/smiles_input" and cashe_smiles == "":
        smis = search.split("?smiles=")[-1]
        smis = url_smiles_to(smis)
        if cashe_smiles == smis:
            return no_update, no_update, no_update

        return smis, "", smis

    if n_clicks is None and smis=="":
        return no_update, no_update, no_update

    smis = value.replace("\n", ",")
    smis = smis.replace(".", ",") #ChemDrawから複数化合物のSMILES用。
    if cashe_smiles == smis:
            return no_update, no_update, no_update

    return smis, "", smis


#save アドレス取得　⇒　save_htmlへcallback連鎖
@app.callback(
    Output("save_path","data"),
    Output("save_modal", "is_open"),
    Input("save_button", "n_clicks"),
    Input("save_over_button", "n_clicks"),
    Input("saveas_button", "n_clicks"),
    State("user_id", "data"),
    State("save_path","data"),
    prevent_initial_call=True
)
def save_modal_func(save_click, save_over_click, saveas_click, user_id, save_path):
    if not callback_context.triggered:
        return no_update, no_update, no_update, no_update

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0]
    if trigger == "save_button" and save_path is not None:
        return no_update, True
    elif trigger == "save_over_button":
        return save_path, False
    else:
        save_file = "%d_%f"%(user_id, time())
        save_path= hashlib.sha256(save_file.encode()).hexdigest()
        return save_path, False


#save オールページ
@app.callback(
    Output("save_downloader", "data"),
    Output("url", "href"),
    Input("save_path", "data"),
    State("user_id", "data"),
    State("index_html", "children"),
    prevent_initial_call=True
)
def save_html(save_path, user_id, user_page):
    global user_data

    if user_id is None:
        return no_update, no_update
    else:
        pred_df = user_data[user_id]

    if save_path is None:
        return no_update, no_update

    with open(f"{webgui_path}/temp/{save_path}.jlib", "wb") as pf:
        joblib.dump([pred_df,user_page], pf, 3)

    with open(f"{webgui_path}/temp/{save_path}.html", "w") as pf:
        url_hash = urllib.parse.quote(save_path)
        url = f"{MAIN_URL}/save?path={url_hash}"
        pf.write(f'<meta http-equiv="refresh" content="0;URL={url}">')

    return dcc.send_file(f"{webgui_path}/temp/{save_path}.html"), url


#saveしたページを読みだす
@app.callback(
    Output("index_html", "children"),
    Output("user_id", "data"),
    Output("url","search"),
    Output("url","pathname"),
    Input("url","search"),
    State("url","pathname"),
    State("user_id", "data"),
    prevent_inidital_call = True
)
def load_svae_data(search, pathname, user_id):
    global user_data, user_att_data

    if user_id is not None:
        return no_update, no_update, no_update, no_update

    user_id = get_user_id()
    pred_df = user_data[user_id]
    user_att_data[user_id] = []

    if pathname!="/save":
        return no_update, user_id, no_update, no_update

    save_hash = search.split("?path=")[-1]
    save_hash = urllib.parse.unquote(save_hash)
    save_hash = secure_filename(save_hash)
    if not os.path.exists(f"{webgui_path}/temp/{save_hash}.jlib"):
        return no_update, no_update, no_update, no_update

    with open(f"{webgui_path}/temp/{save_hash}.jlib", "rb") as pf:
        pred_df, user_page = joblib.load(pf)

    user_data[user_id] = pred_df

    return user_page, user_id, search, pathname


#ユーザデータの追加、データ表示するドロップダウンへの追加　⇒ show_columnsを呼び出す
@app.callback(
    Output("pred_others_ok", "n_clicks"),
    Output("results", "children"),
    Output("result_dropdown", "options"),
    Output("result_dropdown", "value"),
    Output("loading_button", "n_clicks"),
    Input({"type":"result", "index":ALL}, "children"),
    State("result_dropdown", "value"),
    State("result_dropdown", "options"),
    State("user_id", "data"),
    State("loading_button", "n_clicks"),
    State("pred_param", "data"),
    prevent_inidital_call = True
)
def show_pred_panel(results, rdd, dropdown_titles, user_id, n_clicks, pred_param):
    global user_data

    if user_id is None:
        return no_update, no_update, no_update, no_update, 1
    else:
        pred_df = user_data[user_id]

    sort=True
    reaction_results = None

    result_idx = -1
    res = [html.Div("", id={"type":"result", "index":i}, style={"display":"none"}) for i in range(4)]

    for i, x in enumerate(results):
        if x == "":
            continue
        result_idx = i
        result_value = x
        break

    if result_idx == -1 : # resultsが全て""の場合
        return no_update, no_update, no_update, no_update, 1
    elif result_idx == 0: # 予測Submit
        sort = False
        smiles = result_value.split(",")
    elif result_idx == 1: # 化学反応API
        result_value = json.loads(result_value)
        reaction_results = pd.DataFrame(result_value)
        smiles = [x for x in reaction_results["smiles"]]
        if len(smiles) == 0:
            return no_update, no_update, no_update, no_update, 1

    else: # それ以外 (部分構造発生、化合物最適化)
        smiles = result_value.split(",")
        smiles = list(set(smiles))
        if len(smiles) == 0:
            return no_update, no_update, no_update, no_update, 1

    col_id = len(dropdown_titles)

    mols = [Chem.MolFromSmiles(x) for x in smiles]
    flag = [x is not None for x in mols]
    mols = [x for x in mols if x is not None]
    smiles = [Chem.MolToSmiles(x) for x in mols]

    if reaction_results is not None:
        reaction_results = reaction_results[flag]
        reaction_results["smiles"]=smiles

    pred_df.append(get_pred_df(smiles, sort=sort, pred_param=pred_param))
    smiles_series = pred_df[col_id]["SMILES"]

    def get_reagent_info(smi):
        if smi in reagent_dict:
            return reagent_dict[smi]
        else:
            return ""

    pred_df[col_id]["id"]=[f"{col_id}-{i}" for i in range(len(smiles_series))]
    pred_df[col_id]["selection"]=[0 for i in range(len(smiles_series))]
    pred_df[col_id]["MW"]=[Descriptors.ExactMolWt(Chem.MolFromSmiles(x)) for x in smiles_series]

    if reaction_results is not None:
        reaction_results = reaction_results.set_index("smiles")
        pred_df[col_id]["reagent-info"] = [reaction_results.loc[x]["reagent-info"] for x in pred_df[col_id]["SMILES"]]

    pred_df[col_id]["memo"] = ["" for i in range(len(smiles_series))]
    user_data[user_id]=pred_df

    rdd.append(col_id)
    dropdown_titles.append({'label': f"{col_id+1}.{OPTION_LABEL[result_idx]}", 'value': col_id})

    return 1, res, dropdown_titles, rdd, 1



#カラムの外枠を追加する。 カラムの追加・リスト・テーブルの切り替えや各カラムの表示・非表示で呼び出される。⇒ show_single_columnへ
@app.callback(
    Output("result_table", "children"),
    Input("result_dropdown", "value"),
    Input("pred_others_ok", "n_clicks"),
    Input("view_tabs", "active_tab"),
    State({"type":"col_tab", "index":ALL}, "active_tab"),
    State("result_dropdown", "options"),
    State("user_id", "data"),
    State("pred_param", "data"),
    prevent_inidital_call = True
)
def show_columns(value, n_clicks, view_mode, col_show_num, options, user_id, pred_param):
    global user_data, time_stamps
    if user_id is None:
        raise PreventUpdate()
    else:
        pred_df = user_data[user_id]

    cols = []

    for i in range(len(options)):
        if i in value:
            smiles = pred_df[i][pred_df[i]["display"]==1]["SMILES"]
            if i >= len(col_show_num):
                cols.append(get_column(len(smiles),i, options[i]["label"], 1, pred_param))
            else:
                cols.append(get_column(len(smiles),i, options[i]["label"], col_show_num[i], pred_param))

    return [dbc.Row(children = cols,id = "output", style={"display":"flex", "flex-wrap":"nowrap"})]


#カラム内の表示を行う。 表示切替で常に呼び出される
@app.callback(
    Output({"type":"sort_div", "index":MATCH}, "children"),
    Output({"type":"col_row_pred", "index":MATCH}, "children"),
    Input({"type":"sort_sub", "index":MATCH}, "children"),
    Input({"type":"col_tab", "index":MATCH, "sub_index":ALL}, "active_tab"),
    State({"type":"col_tab", "index":MATCH, "sub_index":ALL}, "id"),
    State({"type":"sort_param", "index":MATCH}, "value"),
    State("view_tabs", "active_tab"),
    State("user_id", "data"),
    State("pred_others_check", "value"),
    State("use_memo", "value"),
    State("pred_param", "data"),
    prevent_inidital_call = True
)
def show_single_column(message, active_tab, tab_id, sort_param, view_mode, user_id, others, use_memo, pred_param):
    global usr_data, time_stamps
    if user_id is None:
        raise PreventUpdate()
    else:
        pred_df = user_data[user_id]

    if active_tab is None:
        raise PreventUpdate()

    sub_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if sub_id != "" and json.loads(sub_id)["type"]=="col_tab":
        sub_id = json.loads(sub_id)["sub_index"]
    else:
        sub_id=0

    active_tab = active_tab[sub_id]
    tab_id=tab_id[sub_id]

    col_id = tab_id["index"]
    smiles = pred_df[col_id][pred_df[col_id]["display"]==1]["SMILES"]

    end = active_tab * PER_COL
    start = (active_tab-1) * PER_COL

    table = False if (view_mode == "list_view") else True
    use_memo = False if (len(use_memo) == 0) else True

    sort_sub = html.Div("", id={"type":"sort_sub", "index":col_id}, style={"display":"none"})

    if message == "":
        global_update_flag3 = True
    time_stamps[user_id]=time()

    return sort_sub, [get_pred_panel(smiles[start:end], col_id, pred_df[col_id],
                                     table, sort_param, others, use_memo, classes=pred_param["classification"])]


#カラム内のページ管理（カラム内のタブの中身書き換え） *ページマネージャがdbcにあったのでそれを使っても良いかも。
@app.callback(
    Output({"type":"col_tab", "index":MATCH, "sub_index":ALL}, "children"),
    Input({"type":"col_tab", "index":MATCH, "sub_index":ALL}, "active_tab"),
    State({"type":"col_tab", "index":MATCH, "sub_index":ALL}, "id"),
    State({"type":"col_tab", "index":MATCH, "sub_index":ALL}, "children"),
    State("user_id", "data"),
    prevent_inidital_call = True
)
def column_tabs(active_tab, tab_id, children, user_id):
    global user_data
    if user_id is None:
        raise PreventUpdate()
    else:
        pred_df = user_data[user_id]

    trg =callback_context.triggered[0]["prop_id"].split(".")[0]
    if trg != "":
        trg_dict = json.loads(trg)
        trg_type = trg_dict["type"]
        if trg_type == "col_tab":
            sub_id = trg_dict["sub_index"]
            trg_type = ""
        else:
            trg_type = "sort_sub"
            sub_id = 0
    else:
        sub_id=0
        trg_type = ""

    active_tab = active_tab[sub_id]
    tab_id=tab_id[sub_id]
    children = children[sub_id]

    col_id = tab_id["index"]
    smiles = pred_df[col_id][pred_df[col_id]["display"]==1]["SMILES"]
    max_tabs = int(np.ceil(len(smiles)/PER_COL))

    current_min = children[0]["props"]["tab_id"]
    if trg_type == "sort_sub":
        active_tab = 1


    if active_tab % (MAX_TAB-1) == 1 and active_tab != 1 and active_tab != max_tabs:
        if current_min == active_tab:
            start = active_tab - MAX_TAB+1
            end = active_tab+1
        else:
            start = active_tab
            end = active_tab + MAX_TAB
            if end > max_tabs+1:
                end = max_tabs+1
    else:
        raise PreventUpdate()

    return [[dbc.Tab(label=i, tab_id=i) for i in range(start, end)]]*2  #上下にページ用のタブがあるので2つ


#ページ数表示
@app.callback(
    Output({"type":"col_num", "index":MATCH, "sub_index":ALL}, "children"),
    Output({"type":"col_tab", "index":MATCH, "sub_index":ALL}, "active_tab"),
    Input({"type":"col_tab", "index":MATCH, "sub_index":ALL}, "active_tab"),
    Input({"type":"sort_sub", "index":MATCH}, "children"),
    State({"type":"col_tab", "index":MATCH, "sub_index":ALL}, "id"),
    State("user_id", "data"),
    prevent_inidital_call = True
)
def column_page(active_tab, sort_sub, tab_id, user_id):
    global user_data
    if user_id is None:
        raise PreventUpdate()
    else:
        pred_df = user_data[user_id]

    trg =callback_context.triggered[0]["prop_id"].split(".")[0]
    if trg != "":
        trg_dict = json.loads(trg)
        trg_type = trg_dict["type"]
        if trg_type == "col_tab":
            sub_id = trg_dict["sub_index"]
            trg_type = ""
        else:
            trg_type = "sort_sub"
            sub_id = 0
    else:
        sub_id=0
        trg_type = ""

    active_tab = active_tab[sub_id]
    tab_id=tab_id[sub_id]

    col_id = tab_id["index"]
    smiles = pred_df[col_id][pred_df[col_id]["display"]==1]["SMILES"]
    num_smiles = len(smiles)
    max_tabs = int(np.ceil(num_smiles/PER_COL))

    global_update_flag2 = True #show_single_columnの連鎖的なcallbackを禁止（表示が遅くなるため）

    if trg_type == "sort_sub":
        active_tab = 1

    start = (active_tab-1)*PER_COL+1
    end = PER_COL*active_tab if PER_COL*active_tab < num_smiles else num_smiles

    return [f"({active_tab}/{max_tabs} total:{num_smiles}) ",
            f"（compd:{start}-{end}/{num_smiles}, tab:{active_tab}/{max_tabs}）"], [active_tab]*2  #上下にページ用のタブがあるので2つ


#カラム内のソート及び緑以外を非表示
@app.callback(
    Output({"type":"sort_sub", "index":MATCH}, "children"),
    Input({"type":"option_ok", "index":MATCH}, "n_clicks"),
    State({"type":"negative_ascending", "index":MATCH}, "value"),
    State({"type":"sort_param", "index":MATCH}, "value"),
    State({"type":"negative_param", "index":MATCH}, "value"),
    State({"type":"mw_min", "index":MATCH}, "value"),
    State({"type":"mw_max", "index":MATCH}, "value"),
    State({"type":"logd_min", "index":MATCH}, "value"),
    State({"type":"logd_max", "index":MATCH}, "value"),
    State({"type":"range_param", "index":MATCH}, "value"),
    State({"type":"sort_div", "index":MATCH}, "children"),
    State("user_id", "data"),
    prevent_inidital_call = True
)
def options(ok_click, psum, sort, ref, mw_min, mw_max, logd_min, logd_max, range_param, children, user_id):
    if not callback_context.triggered:
        return no_update

    global user_data, time_stamps
    if user_id is None:
        return no_update
    else:
        pred_df = user_data[user_id]

    sub_id = callback_context.triggered[0]["prop_id"].split(".")[0]

    cid = children["props"]["id"]["index"]

    smiles = pred_df[cid]["SMILES"]
    max_tabs = int(np.ceil(len(smiles)/PER_COL))

    sort_param = []
    ascending=False
    if sort != "" and sort is not None:
        if sort+"_p" in pred_df[cid].columns and "sort_values" not in psum:
            sort_param.append(sort+"_p")
        else:
            sort_param.append(sort)
            if "ascending" in psum:
                ascending=True

    if "better" in psum:
        sort_param.append("sum")

    if len(sort_param)>0:
        pred_df[cid] = pred_df[cid].sort_values(by=sort_param, ascending=ascending)

    if len(ref) != 0:
        disp = np.array(pred_df[cid][[x+"_p" for x in ref]].sum(1)) == len(ref)*2
        disp = disp.astype(np.int64)
        pred_df[cid]["display"]=disp
    else:
        pred_df[cid]["display"]=[1]*len(pred_df[cid])

    disps = []
    if mw_min is not None:
        print(mw_min, pred_df[cid]["MW"])
        disp = np.array(pred_df[cid]["MW"]) >= mw_min
        disps.append(disp.astype(np.int64))

    if mw_max is not None:
        print(mw_max)
        disp = np.array(pred_df[cid]["MW"]) <= mw_max
        disps.append(disp.astype(np.int64))

    if logd_min is not None and range_param is not None:
        print(logd_min)
        disp = np.array(pred_df[cid][range_param]) >= logd_min
        disps.append(disp.astype(np.int64))

    if logd_max is not None and range_param is not None:
        print(logd_max)
        disp = np.array(pred_df[cid][range_param]) <= logd_max
        disps.append(disp.astype(np.int64))

    if len(disps) != 0:
        disp = np.sum(np.stack(disps), axis = 0) + np.array(pred_df[cid]["display"])
        disp = (disp == len(disps)+1).astype(np.int64)
        pred_df[cid]["display"] = disp

    time_stamps[user_id] = time()

    return "OK"


#表示されたソート用Popoverの非表示化
@app.callback(
    Output({"type":"option_popover", "index":MATCH}, "is_open"),
    Input({"type":"option_ok", "index":MATCH}, "n_clicks"),
    Input({"type":"option_cancel", "index":MATCH}, "n_clicks")
)
def option_popover_open(ok_click, cancel_click):
    return False


#表示されたダウンロード用Popoverの非表示化
@app.callback(
    Output({"type":"download_pop", "index":MATCH}, "is_open"),
    Input({"type":"download_ok", "mode":ALL, "index":MATCH}, "n_clicks"),
    Input({"type":"download_cancel", "index":MATCH}, "n_clicks")
)
def download_open(ok_click, cancel_click):
    return False


#ダウンロード処理（カラム内・選択の両方）
@app.callback(
    Output({"type":"downloader", "index":MATCH}, "data"),
    Input({"type":"download_ok", "mode":ALL, "index":MATCH}, "n_clicks"),
    State("result_dropdown", "options"),
    State("user_id", "data"),
    State("pred_others_check", "value"),
    prevent_inidital_call = True
)
def download_click(ok_click, options, user_id, others):
    global user_data, time_stamps
    if user_id is None:
        raise PreventUpdate()
    else:
        pred_df = user_data[user_id]

    sub_id = callback_context.triggered[0]["prop_id"].split(".")[0]
    if sub_id != "":
        j = json.loads(sub_id)
        cid = j["index"]
        filetype_mode = j["mode"]
        if cid == 999:
            target_df = pd.concat(pred_df)
            fname = "whole"
        else:
            target_df = pred_df[cid]
            fname = options[cid]["label"]
    else:
        raise PreventUpdate()

    download_df = target_df[target_df["display"]==1].reset_index(drop=True)

    if sum(download_df["selection"])!=0:
        download_df = download_df[download_df["selection"]==1]

    download_df=download_df[["SMILES"]+[x for x in others+["memo","reagent-info"]]]
    time_stamps[user_id]=time()

    if filetype_mode == "sdf":
        PandasTools.AddMoleculeColumnToFrame(download_df,"SMILES")
        PandasTools.WriteSDF(download_df, f"temp/{fname}.sdf", properties=others)

        return dcc.send_file(f"temp/{fname}.sdf")

    elif filetype_mode == "csv":
        return dcc.send_data_frame(download_df.to_csv, f"{fname}.csv", index=False)
    elif filetype_mode == "excel":
        return dcc.send_data_frame(download_df.to_excel, f"{fname}.xlsx", sheet_name=f"{fname}", index=False)


#ドッキング処理 (化合物画像をクリックするとDockingとView Selectionsボタンのリンクを変更する）
@app.callback(
    Output("new_tab", "href"),
    Input( {"type":"ImgPanel", "index":ALL}, "children"),
    State("user_id", "data"),
    prevent_initial_call=True
)
def docking_click(children, user_id):
    global user_data, time_stamps
    if user_id is None:
        raise PreventUpdate()
    else:
        pred_df = user_data[user_id]

    target_df = pd.concat(pred_df)
    download_df = target_df.reset_index(drop=True)

    if sum(download_df["selection"])==0:
        return no_update

    download_df = download_df[download_df["selection"]==1]

    smiles = [x for x in download_df["SMILES"]]
    smiles = get_url_smiles(smiles)
    new_tab_url = MAIN_URL +"/smiles_input?smiles=" + smiles

    time_stamps[user_id]=time()

    return new_tab_url


#化合物選択用（化合物画像をクリックするとborderをgrooveかnoneに切り替える）
@app.callback(
    Output( {"type":"ImgPanel", "index":MATCH}, "children"),
    Input( {"type":"ImgPanel", "index":MATCH}, "n_clicks"),
    State( {"type":"ImgPanel", "index":MATCH}, "children"),
    State("user_id", "data"),
    prevent_initial_call=True
)
def image_click(n_clicks, children, user_id):
    global user_data

    if user_id is None:
        raise PreventUpdate()
    else:
        pred_df = user_data[user_id]

    td = json.loads(callback_context.triggered[0]["prop_id"].split(".")[0])
    index = td["index"]
    col_id = int(index.split("-")[0])
    num_id = int(index.split("-")[1])

    style = children["props"]["style"]
    if "border" not in style.keys():
        style["border"] = "solid"
        style["color"] = "#ffffff10"

    if style["border"]=="groove":
        style["border"]="solid"
        style["color"] = "#ffffff10"
        smi = pred_df[col_id][pred_df[col_id]["id"]==index]["SMILES"]
        pred_df[col_id].loc[smi, "selection"] = 0
    else:
        style["border"]="groove"
        style["color"] = "#ffffff10"
        smi = pred_df[col_id][pred_df[col_id]["id"]==index]["SMILES"]
        pred_df[col_id].loc[smi, "selection"] = 1

    user_data[user_id] = pred_df

    children["props"]["style"]=style

    return children


# 化合物が１つ以上選択されたら選択化合物用のDownload及びDockingボタンと選択解除ボタンが表示される
@app.callback(
    Output({"type":"download_button", "index":999}, "style"),
    Output("whole_clear", "style"),
    Output("new_tab", "style"),
    Input( {"type":"ImgPanel", "index":ALL}, "children"),
    Input( {"type":"ImgBlock", "index":ALL}, "style"),
    State({"type":"download_button", "index":999}, "style"),
    State("whole_clear", "style"),
    State("new_tab", "style"),
    prevent_initial_call=True
)
def show_button( children, img_style, download_style, clear_style, new_tab_style):
    flag = False

    for x in children:
        style = x["props"]["style"]
        if "border" not in style.keys():
            continue

        if style["border"]=="groove":
            flag=True
            break

    if flag:
        download_style["display"]="inline-block"
        clear_style["display"]="inline-block"
        new_tab_style["display"]="inline-block"
    else:
        download_style["display"]="none"
        clear_style["display"]="none"
        new_tab_style["display"]="none"

    return download_style, clear_style, new_tab_style


#まとめて選択解除するボタンの処理
@app.callback(
    Output({"type":"ImgBlock", "index":ALL}, "style"),
    Input("whole_clear", "n_clicks"),
    State({"type":"ImgBlock", "index":ALL}, "style"),
    prevent_initial_call=True
)
def clear_selections(n_clicks, styles):
    for i,x in enumerate(styles):
        styles[i]["border"]="solid"
        styles[i]["border-color"]="#ffffff10"

    return styles


# Prediction othersのPopoverを閉じる
@app.callback(
    Output("pred_others_pop", "is_open"),
    Output("pred_others_check", "value"),
    Input("pred_others_ok", "n_clicks"),
    Input("pred_others_cancel", "n_clicks"),
    State("pred_others_store", "data"),
    prevent_initial_call=True
)
def close_pred_others_pop(ok, cancel, value):
    cid = callback_context.triggered[0]["prop_id"].split(".")[0]
    if cid == "pred_others_ok":
        return False, no_update
    else:
        return False, value


# Prediction othersのPopoverの開いた時の情報を保持閉
@app.callback(
    Output("pred_others_store", "data"),
    Input("pred_others_pop", "is_open"),
    State("pred_others_check", "value"),
    prevent_initial_call=True
)
def memory_pred_others_check(is_open, value):
    if is_open:
        return value
    else:
        return no_update


# 注目構造の表示とweightの取得
@app.callback(
    Output("attention_check", "value"),
    Input("attention_check", "value"),
    State("user_id", "data"),
    State("pred_others_check", "value"),
    prevent_initial_call=True
)
def attention_popover(value, user_id, others):

    global user_data, user_att_data
    if user_id is None:
        raise PreventUpdate()
    else:
        pred_df = user_data[user_id]
        att_data = user_att_data[user_id]

    if len(value)==0 or len(pred_df)==0:
        return value

    for i in range(len(att_data), len(pred_df)):
        tdf = pred_df[i].sort_values("id")
        res = get_weight(list(tdf["SMILES"]))
        if res is not None:
            att_data.append(res)

    user_att_data[user_id] = att_data

    return value


# Popoverの中身の表示と削除 Open時に中身を表示、close時に中身を削除 （ユーザー側のデータ量を減らすため）
@app.callback(
    Output({"type":"pred_hover", "param":MATCH, "index":MATCH}, "children"),
    Output({"type":"pred_hover", "param":MATCH, "index":MATCH}, "style"),
    Input({"type":"pred_hover", "param":MATCH, "index":MATCH}, "is_open"),
    State("user_id", "data"),
    State("attention_check", "value"),
    prevent_initial_call=True
)
def attention_pop_draw(is_open, user_id, value):
    if len(value)==0:
        return no_update, no_update

    if is_open == False:
        return [], {"display":"none"}

    global user_data, user_att_data
    if user_id is None:
        raise PreventUpdate()
    else:
        pred_df = user_data[user_id]
        att_data = user_att_data[user_id]

    td = json.loads(callback_context.triggered[0]["prop_id"].split(".")[0])
    param = td["param"].replace("___", ".")
    index = td["index"]
    col_id = int(index.split("-")[0])
    num_id = int(index.split("-")[1])

    if col_id > len(att_data)-1:
        return no_update, no_update

    tdf = pred_df[col_id]
    smi = tdf[tdf["id"]==index]
    smi = list(smi["SMILES"])[0]

    weight = att_data[col_id][param][num_id]

    #溶解度に関しては値が大きくなる方が青表示
    if param == "Solubility(ug/mL)":
        weight = [-w for w in weight]
        img = chem_util.getWeightImg(smi,weight)
    else:
        img = chem_util.getWeightImg(smi,weight)

    return html.Img(src=img), {"display":""}


# メモのInputを表示する。
@app.callback(
    Output({"type":"memo", "index":ALL}, "style"),
    Input("use_memo", "value"),
    State({"type":"memo", "index":ALL}, "style"),
    prevent_initial_call=True
)
def show_memo(value, styles):
    if len(value)==0:
        return [{"display":"none"} for x in range(len(styles))]
    else:
        return [{"display":""} for x in range(len(styles))]


#メモデータ保存用
@app.callback(
    Output( {"type":"memo", "index":MATCH}, "value"),
    Input( {"type":"memo", "index":MATCH}, "value"),
    State("user_id", "data"),
    prevent_initial_call=True
)
def save_memo(value, user_id):
    global user_data

    if user_id is None:
        return no_update
    else:
        pred_df = user_data[user_id]

    td = json.loads(callback_context.triggered[0]["prop_id"].split(".")[0])
    index = td["index"]
    col_id = int(index.split("-")[0])
    num_id = int(index.split("-")[1])

    smi = pred_df[col_id][pred_df[col_id]["id"]==index]["SMILES"]
    pred_df[col_id].loc[smi, "memo"] = value
    user_data[user_id] = pred_df

    return no_update



#loading modalの表示・非表示
@app.callback(
    Output("loading_modal", "is_open"),
    Input({'type': 'panel_linker_func', 'index': "hopping"}, "n_clicks"),
    Input({'type': 'panel_linker_func', 'index': "growing"}, "n_clicks"),
    Input({"type": "panel_react_func", "index": ALL}, "n_clicks"),
    Input({"type": "panel_opt1_func", "index": ALL}, "n_clicks"),
    Input("loading_button", "n_clicks"),
    prevent_initial_call=True
)
def loading_modal_open(hoping, growing, react, opt, loading_button):
    if not callback_context.triggered:
        return False

    if len(callback_context.triggered)!=1:
        return False

    trigger = callback_context.triggered[0]["prop_id"].split(".")[0]

    if str(trigger) == "loading_button":
        return False

    open_type = ["panel_linker_func", "panel_react_func", "panel_opt1_func"]
    open_index = ["hopping", "linking", "growing", "submit"]
    trigger_dict = ast.literal_eval(trigger)
    if trigger_dict["type"] not in open_type:
        return False

    if trigger_dict["index"].split("_")[0] in open_index:
        return True

    return False



@app.callback(
    Output({"type": "param-hover-body", "index": ALL}, "children"),
    Input({"type": "param-hover", "index": ALL}, "is_open"))
def hover_test(n_clicks):
    #print(type(callback_context.outputs_list), )
    #print(callback_context.outputs_list)
    outputs_len = len(callback_context.outputs_list)
    res = [html.Img(style={"width": "250px", "height": "250px"}) for i in range(outputs_len)]

    if (callback_context.triggered[0]["prop_id"] == "."):
        return res

    trigger_dict = ast.literal_eval(callback_context.triggered[0]["prop_id"].split(".")[0])
    #print(trigger_dict)

    i, param, smi = trigger_dict["index"].split("_")

    idx = 0
    for j in range(len(res)):
        if callback_context.outputs_list[j]["id"]["index"] == f"{i}_{param}":
            idx = j
            break

    if param in param_dict:
        param = param_dict[param]

    #print(param)

    b64 = chem_util.getWeightImgBase64FromAPI2(smi, param)
    res[j] = html.Img(src=f"data:image/png;base64,{b64}", style={"width": "250px", "height": "250px"})

    #print(res)

    return res


@app.callback(
    Output({"type": "edit_hash", "index": MATCH}, "value"),
    Output({"type": "edit", "index": MATCH}, "href"),
    Input({"type": "edit_popover", "index": MATCH}, "is_open"),
    State("user_id", "data"),
    prevent_initial_call=True
)
def edit_chem(is_open, user_id):
    global user_data

    if not is_open:
        return no_update, no_update

    if user_id is None:
        return no_update, no_update
    else:
        pred_df = user_data[user_id]

    td = json.loads(callback_context.triggered[0]["prop_id"].split(".")[0])
    index = td["index"]
    col_id = int(index.split("-")[0])
    num_id = int(index.split("-")[1])

    smi = pred_df[col_id][pred_df[col_id]["id"]==index]["SMILES"].values[0]
    smi = get_url_smiles([smi])

    #POST先URL
    mhash = hashlib.md5(smi.encode()).hexdigest()
    url = f"{CHEMDOODLE_API_URL}/make?smiles={smi}&hash={mhash}"

    return mhash, url


@app.callback(
    Output({"type": "edit_popover", "index": MATCH}, "is_open"),
    Input({"type": "edit_ok", "index": MATCH}, "n_clicks"),
    Input({"type": "edit_cancel", "index": MATCH}, "n_clicks"),
    prevent_initial_call=True
)
def edit_okcancel_close(on, cn):
    return False


@app.callback(
    Output({"type": "WholePanel", "index": MATCH}, "children"),
    Output({"type": "smi_input", "index": MATCH}, "value"),
    Input({"type": "edit_ok", "index": MATCH}, "n_clicks"),
    State({"type": "edit_hash", "index": MATCH}, "value"),
    State("pred_others_check", "value"),
    State("user_id", "data"),
    State("use_memo", "value"),
    State("pred_param", "data"),
    prevent_initial_call=True
)
def edit_state(n_clicks, mhash, others, user_id, use_memo, pred_param):
    global user_data

    if n_clicks==0:
        return no_update, no_update

    if user_id is None:
        return no_update, no_update
    else:
        pred_df = user_data[user_id]

    if mhash is None:
        return no_update, no_update

    td = json.loads(callback_context.triggered[0]["prop_id"].split(".")[0])
    index = td["index"]
    col_id = int(index.split("-")[0])
    num_id = int(index.split("-")[1])
    pre_smi = pred_df[col_id][pred_df[col_id]["id"]==index]["SMILES"].values[0]

    path = get_file_path(mhash)
    smi = [Chem.MolToSmiles(Chem.MolFromMolFile(path))]
    #os.remove(fname)

    smi = chem_util.getValidSmiles(smi)
    pred_df_=get_pred_df(smi, pred_param=pred_param)
    smiles_series = pred_df_["SMILES"]

    pred_df_["id"]=[index]
    pred_df_["selection"]=[0]
    pred_df_["MW"]=[Descriptors.ExactMolWt(Chem.MolFromSmiles(x)) for x in smiles_series]
    pred_df_["reagent-info"] = [""]
    pred_df_["memo"] = [""]

    #pred_dfへインサート
    for x in pred_df_.columns:
        pred_df[col_id].loc[pre_smi,x]=pred_df_.iloc[0][x]

    pred_df[col_id] = pred_df[col_id].reset_index(drop=True)
    pred_df[col_id] = pred_df[col_id].set_index("SMILES", drop=False)
    pred_df[col_id] = pred_df[col_id].drop_duplicates("SMILES")

    user_data[user_id]=pred_df

    res = [
        html.Td(getImgPanel(index, smi[0]),
                id = {"type":"ImgPanel", "index":index}),
        html.Td(get_pred_panel_list(index, smi[0], pred_df_.iloc[0].to_dict(),
                                    others, use_memo, pred_df_.iloc[0]["memo"], pred_param["classification"]),
                id = {"type":"PredPanel", "index":index}),
        html.Td(getFuncPanel(index, smi[0]),
                id = {"type":"FuncPanel", "index":index})
    ]

    return res, chem_util.getKekuleSmiles(smi[0])


@app.callback(
    Output("smiles_input_form", "style"),
    Output("menu_smiles_input", "children"),
    Input("menu_smiles_input", "n_clicks"),
    State("smiles_input_form", "style"),
    prevent_initial_call=True
)
def smiles_input_open_close(n_clicks, style):
    if style is None:
        return no_update

    if style == {"display":"none"}:
        return {"position": "relative", "left": "25%"}, "Unshow"
    else:
        return {"display":"none"}, "Show"

