from dash import dcc, html
import dash_bootstrap_components as dbc
import json
import pandas as pd
import requests
import color
import numpy as np
from tqdm import tqdm
from time import time

import panel

from constants import PER_COL, MAX_TAB, PREDICTION_API_URL, PANEL_TABLE_STYLE, SELECTION_BORDER_STYLE, CHEMDOODLE_API_URL


def get_child(children, child_id):
    if type(children) is list:
        for child in children:
            ret = get_child(child, child_id)
            if ret is not None:
                return ret

    elif type(children) is dict:
        if "id" in children["props"].keys():
            if children["props"]["id"] == child_id:
                return children
            else:
                if "children" in children["props"].keys():
                    return get_child(children["props"]["children"], child_id)
                else:
                    return None
        else:
            if "children" in children["props"].keys():
                return get_child(children["props"]["children"], child_id)
            else:
                return None

    else:
        return None


def sort_items(sort_values, items, sort_reverse):
    si = [sort_values, list(range(len(items)))]
    si = list(map(list, zip(*si)))
    si = sorted(si, reverse = sort_reverse)
    value, sorted_id = list(map(list, zip(*si)))
    items = [items[x] for x in sorted_id]

    return items


def get_pred_df(smiles, base_df=None, sort=True, pred_param=[]):
    comma_smiles = ",".join(smiles)
    pred_r = requests.get(f"{PREDICTION_API_URL}/predict",
                          params={
                              "smiles": comma_smiles,
                          }, proxies={"http": None}, verify=False)

    df = pd.DataFrame(pred_r.json())
    df["SMILES"] = smiles

    if base_df is not None:
        base_df = base_df.reset_index(drop=True)
        df = pd.merge(base_df, df, how="outer")

    df = df.drop_duplicates("SMILES")
    df = df.set_index("SMILES", drop=False)
    df["display"] = [1]*len(df)
    df = get_point(df, pred_param)
    if sort:
        df = df.sort_values(by="sum", ascending=False)

    return df


def get_param():
    pred_r = requests.get(f"{PREDICTION_API_URL}/params",
                          proxies={"http": None}, verify=False)

    params = pred_r.json()
    return params


def get_file_path(mhash):
    pred_r = requests.get(f"{CHEMDOODLE_API_URL}/file",
                          params={"hash": mhash},
                          proxies={"http": None}, verify=False)
    path = pred_r.text
    return path


def get_weight(smiles):
    batch = 300
    times = int(np.ceil(len(smiles)/batch))

    results = []
    for i in range(times):
        comma_smiles = ",".join(smiles[i*batch:(i+1)*batch])
        results.append( requests.get(f"{PREDICTION_API_URL}/weight",
                                     params={
                                         "smiles": comma_smiles,
                                     }, proxies={"http": None}, verify=False).json())
    if len(results) ==0:
        return None
    else:
        res = {}
        for i,x in enumerate(results[0].keys()):
            ps = []
            for y in results:
                ps.extend(y[x])
            res[x]=ps

    return res


def get_url_smiles(smiles):
    return ",".join(smiles).replace("%", "%25").replace("#", "%23").replace("+", "%2B")

def url_smiles_to(smiles):
    return smiles.replace("%25", "%").replace("%23", "#").replace("%2B", "+")


#総合評価計算用
def get_point(df, pred_param):
    for x in pred_param["classification"]:
        df[x + "_p"] = np.array(df[x] <= 0.4, dtype=np.float64)
        df[x + "_p"] += np.array(df[x] < 0.6, dtype=np.float64)
    df["sum"] = df[[x+"_p" for x in pred_param["classification"] if x != "SMILES"]].sum(1)

    return df


def get_neg_num(pred_val_list):
    res = []
    param_list = "Ames,CYP1A2,CYP2C19,CYP2C9,CYP2D6,CYP3A4,Cyto_10uM,Cyto_50uM,hERG_10uM,hERG_1uM,Pgp".split(",")

    return [sum([l[p] == "negative" for p in param_list]) for l in pred_val_list]


def get_pred_panel(smiles, column_id, pred_df, table=False,
                   param="", others=[], use_memo=False, classes=[]):#, use_func_button=True, sort_flag=False):
    compound_panel_list = []
    for i, smi in enumerate(smiles):
        pred_val = pred_df.loc[smi]
        border = SELECTION_BORDER_STYLE[pred_df.loc[smi]["selection"]]
        idx = pred_df.loc[smi]["id"]
        if table:
            temp = panel.getTablePanel(idx, smi, pred_val, param, others,
                                       border, use_memo, pred_df.loc[smi,"memo"], classes)
        else:
            temp = panel.getWholePanel(idx, smi, pred_val, others,
                                       border, use_memo, pred_df.loc[smi,"memo"], classes)#,use_func_button=use_func_button)

        compound_panel_list.append(temp)

    table_body = [html.Tbody(compound_panel_list)]

    if table:
        res = html.Div(compound_panel_list, style={"display":"flex","flex-wrap":"wrap"})
    else:
        res = dbc.Table(table_body,
                  bordered=True,
                  style=PANEL_TABLE_STYLE,
                  id=f"list_{column_id}")
    return res


def get_column(num_smiles, col_id, option_label, active_tab = 1, pred_param=[]):
    num_col = int(np.ceil(num_smiles/PER_COL))
    len_col = num_col
    if num_col > MAX_TAB:
        num_col = MAX_TAB

    start = PER_COL*(active_tab-1)+1
    end = PER_COL*active_tab if PER_COL*active_tab < num_smiles else num_smiles

    col = dbc.Col(
        [
            dbc.Row([
                html.Pre(f"{option_label} ", id="column_title",
                         style={"font-size":"large", "color":"#0000ff"}),
                html.Pre(f"({active_tab}/{len_col}) ",
                         id={"type":"col_num", "index":col_id, "sub_index":0},
                         style={"font-size":"large", "color":"#4444ff"}),
                dbc.Tabs(
                    [dbc.Tab(label=i+1, tab_id=i+1) for i in range(num_col)],
                    id={"type":"col_tab", "index":col_id, "sub_index":0},
                    style={"font-size":"small"},
                    active_tab=active_tab,
                ),
                html.Pre("  "),
                get_option(col_id, pred_param),
                get_downloader(col_id)
            ]),
            dbc.Row(id={"type":"col_row_pred", "index":col_id}),
            dbc.Row([
                html.Pre(f"({start}-{end}/{num_smiles}) ",
                         id={"type":"col_num", "index":col_id, "sub_index":1},
                         style={"font-size":"large", "color":"#4444ff"}),
                dbc.Tabs(
                    [dbc.Tab(label=i+1, tab_id=i+1) for i in range(num_col)],
                    id={"type":"col_tab", "index":col_id, "sub_index":1},
                    style={"font-size":"small"},
                    active_tab=active_tab,
                ),
            ])
        ],
        id={"type":"result_column", "index":col_id})

    return col


def get_option(col_id, pred_param):
    layout = html.Div(
        [
            html.Div(html.Div("", id={"type":"sort_sub", "index":col_id}, style={"display":"none"}), id = {"type":"sort_div", "index":col_id}),
            dbc.Button("Sort Menu", id = {"type":"option_button", "index":col_id}, style={}, size="sm",
                      outline=True, color="primary", className="me-1"),
            dbc.Popover(
                [
                    html.Div("Sort", style={"font-size":"large", "color":"#4444ff"}),
                    dcc.Checklist(
                        options=[
                            {'label': 'Classification score', 'value': 'better'},
                            {'label': 'Use value ', 'value': 'sort_values'},
                            {'label': 'Ascending ', 'value': 'ascending'},
                        ],
                        value=['better'],
                        id={"type":"negative_ascending", "index":col_id},
                        style={"font-size":"medium"}
                    ),
                    dcc.Dropdown(
                        id={"type":"sort_param", "index":col_id},
                        options=[{"label": x, "value": x} for x in pred_param["all"]],
                        value=""
                    ),
                    html.Br(),
                    html.Div("Search: Green panel only", style={"font-size":"large", "color":"#4444ff"}),
                    dcc.Dropdown(
                        id={"type":"negative_param", "index":col_id},
                        options=[{"label": x, "value": x} for x in pred_param["classification"]],
                        value=[],
                        multi=True
                    ),
                    html.Br(),
                    html.Div("Range: Value", style={"font-size":"large", "color":"#4444ff"}),
                    html.Table([
                        html.Tr([html.Td("Index"),html.Td(": "),
                                 html.Td("Over", style={"text-align":"center"}),
                                 html.Td("Under", style={"text-align":"center"})]),
                        html.Tr([html.Td("Mol weight", style={"width":"90px"}),html.Td(":", style={"width":"20px"}),
                                html.Td(dcc.Input(id={"type":"mw_min", "index":col_id},
                                                  type="number", style={"width":"70px"})),
                                html.Td(dcc.Input(id={"type":"mw_max", "index":col_id},
                                                  type="number", style={"width":"70px"}))]),
                        html.Tr([html.Td(dcc.Dropdown(id={"type":"range_param", "index":col_id},
                                                      options=[{"label": x,
                                                                "value": x} for x in pred_param["regression"]]),
                                         style={"width":"90px"}),html.Td(":", style={"width":"20px"}),
                                html.Td(dcc.Input(id={"type":"logd_min", "index":col_id}, type="number",
                                                  style={"width":"70px"})),
                                html.Td(dcc.Input(id={"type":"logd_max", "index":col_id}, type="number",
                                                  style={"width":"70px"}))]),
                    ]),
                    html.Br(),
                    html.Div([
                        dbc.Button("OK", id = {"type":"option_ok", "index":col_id},
                                   n_clicks=0, style={"width":"50%"}),
                        dbc.Button("Cancel", id = {"type":"option_cancel", "index":col_id},
                                   n_clicks=0, style={"width":"50%"}),
                    ])
                ],
                target = {"type":"option_button", "index":col_id},
                trigger="legacy",
                style = {"height":"480px", "width":"500px"},
                id =  {"type":"option_popover", "index":col_id}
            )
        ]
    )

    return layout

def get_downloader(col_id, display="inline-block", size="sm"):
    layout = html.Div(
        [
            dbc.Button("Download", id = {"type":"download_button", "index":col_id}, style={"display":display}, size=size,
                      outline=True, color="primary", className="me-1"),
            dcc.Download(id={"type":"downloader", "index":col_id}),
            dbc.Popover(
                [
                    html.Div("Dowload format", style={"margin": "10px"}),
                    html.Div([
                        dbc.Button("SDF", style={"width": "100px"}, n_clicks=0, id={"type":"download_ok", "mode": "sdf", "index":col_id}),
                        html.Div("", style={"height": "1px"}),
                        dbc.Button("CSV", style={"width": "100px"}, n_clicks=0, id={"type":"download_ok", "mode": "csv", "index":col_id}),
                        html.Div("", style={"height": "1px"}),
                        dbc.Button("Excel", style={"width": "100px"}, n_clicks=0, id={"type":"download_ok", "mode": "excel", "index":col_id}),
                        html.Div("", style={"height": "5px"}),
                        dbc.Button("Cancel", outline="True", color="primary", style={"width": "100px"},
                                   n_clicks=0, id = {"type":"download_cancel", "index":col_id}),
                    ], style={"margin": "10px", "text-align": "center"}),

                ],
                target = {"type":"download_button", "index":col_id},
                trigger="legacy",
                style = {"height":"280px", "width":"200px"},
                id =  {"type":"download_pop", "index":col_id}
            )
        ]
    )

    return layout


def get_downloader_menu(col_id, display="inline-block", size="sm"):
    layout = html.Div(
        [
            dbc.DropdownMenuItem("Download", id = {"type":"download_button", "index":col_id}, style={"display":display}),
            dcc.Download(id={"type":"downloader", "index":col_id}),
            dbc.Popover(
                [
                    html.Div("Download format", style={"margin": "10px"}),
                    html.Div([
                        dbc.Button("SDF", style={"width": "100px"}, n_clicks=0, id={"type":"download_ok", "mode": "sdf", "index":col_id}),
                        html.Div("", style={"height": "1px"}),
                        dbc.Button("CSV", style={"width": "100px"}, n_clicks=0, id={"type":"download_ok", "mode": "csv", "index":col_id}),
                        html.Div("", style={"height": "1px"}),
                        dbc.Button("Excel", style={"width": "100px"}, n_clicks=0, id={"type":"download_ok", "mode": "excel", "index":col_id}),
                        html.Div("", style={"height": "5px"}),
                        dbc.Button("Cancel", outline="True", color="primary", style={"width": "100px"},
                                   n_clicks=0, id = {"type":"download_cancel", "index":col_id}),
                    ], style={"margin": "10px", "text-align": "center"}),

                ],
                target = {"type":"download_button", "index":col_id},
                trigger="hover",
                style = {"height":"280px", "width":"200px"},
                id =  {"type":"download_pop", "index":col_id}
            )
        ]
    )

    return layout
