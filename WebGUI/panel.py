from dash import dcc, html
import dash_bootstrap_components as dbc
import itertools

import chem_util
import numpy as np
from color import get_val_color
from rdkit import Chem
from rdkit.Chem import Descriptors
import utils
import hashlib

from constants import PADDING_REM, PANEL_STYLE, FUNCTION_BUTTONS, FUNCTION_STYLES, PANEL_TABLE_STYLE, TABLE_MENU_STYLES, CHEMDOODLE_API_URL


def getWholePanel(i, smi, pred_val, params=[], border="solid", use_memo=False, memo="", classes=[]):

    res = html.Tr([
            html.Td(getImgPanel(i, smi, border),
                    id = {"type":"ImgPanel", "index":i}),
            html.Td(get_pred_panel_list(i, smi, pred_val, params, use_memo, memo, classes),
                    id = {"type":"PredPanel", "index":i}),
            html.Td(getFuncPanel(i, smi),
                    id = {"type":"FuncPanel", "index":i})
    ], id = {"type":"WholePanel", "index":i})

    return res


def get_pred_panel_list(i, smi, pred_val, params=[], use_memo=False, memo="", classes=[]):
    pred_panel_list = [getPredPanel(i, smi, pred_val, params, use_memo, memo, classes)]
    if "reagent-info" in pred_val:
        if type(pred_val["reagent-info"]) is str:
            r = pred_val["reagent-info"]
            pred_panel_list.append(
                html.Div(f"reagent-info: {r}")
            )
        else:
            for r in pred_val["reagent-info"]:
                pred_panel_list.append(
                    html.Div(f"reagent-info: {r}")
                )

    return pred_panel_list


def getTablePanel(i, smi, pred_val, param = "", params=[], border="solid", use_memo=False, memo="", classes=[]):
    pred_panel_list = [getPredPanel(i, smi, pred_val, params, use_memo, memo, classes)]
    if "reagent-info" in pred_val:
        if type(pred_val["reagent-info"]) is str:
            r = pred_val["reagent-info"]
            pred_panel_list.append(
                html.Div(f"reagent-info: {r}")
            )
        else:
            for r in pred_val["reagent-info"]:
                pred_panel_list.append(
                    html.Div(f"reagent-info: {r}")
                )

    children = [
                html.Td(getImgPanel(i, smi, border),
                        id = {"type":"ImgPanel", "index":i}),
                html.Td(pred_panel_list,
                        id = {"type":"PredPanel", "index":i}),
                html.Td(getFuncPanel(i, smi),
                        id = {"type":"FuncPanel", "index":i})
    ]

    if param != "":
        param_value =pred_val[param]

        if type(param_value) == str:
            param_value = html.Div(param_value, style={"background":get_val_color(param_value, param, classes)})
        else:
            param_value = html.Div("%.2f"%param_value, style={"background":get_val_color(param_value, param, classes)})

    else:
        param_value =  html.Div("")


    res = html.Div([
        dbc.Button(
            [children[0], param_value],
            outline=True,
            color="dark"
        ),
        dbc.Popover(
            children[1],
            target= {"type":"ImgSvg", "index":i},
            trigger="hover",
            style={"border":"solid", "border-color":"white"}
        ),
        dbc.Popover(
            children[2],
            target= {"type":"FuncPanel", "index":i},
            trigger="legacy",
            style={"width":"120px"},
            id = {"type":"FuncPanelPopover", "index":i}
        )
    ])

    return res



def getImgPanel(i, smi, border="solid"):
    kekuleSmiles = chem_util.getKekuleSmiles(smi)
    url_smi = utils.get_url_smiles([smi])
    mhash = hashlib.md5(smi.encode()).hexdigest()
    mw = Descriptors.ExactMolWt(Chem.MolFromSmiles(smi))
    res = html.Div([
        html.Img(src=chem_util.getSVGFromMol(smi), id ={"type":"ImgSvg", "index":i}, title="ID : %s\nMW : %d"%(i,int(mw))),
        dbc.Popover(
            [dbc.Button("Edit",id={'type': 'edit', 'index': i},
                        href=f"{CHEMDOODLE_API_URL}",
                        target="_blank", n_clicks=0),
             dbc.Button("OK",id={'type': 'edit_ok', 'index': i}, n_clicks=0),
             dbc.Button("Cancel",id={'type': 'edit_cancel', 'index': i}, n_clicks=0),
             #dbc.Button("Docking", href=f"{DOCK_WEB_URL}?smiles={url_smi}", target="_blank"),
             dcc.Input(style={"display": "none"}, id={'type': 'edit_hash', 'index': i})],
            target= {"type":"ImgSvg", "index":i},
            trigger="legacy",
            style={"border":"solid", "border-color":"white"},
            id={'type': 'edit_popover', 'index': i}
        ),
        dcc.Clipboard(target_id={'type': 'smi_input', 'index': i}, style={"color":"black"}),
        dcc.Input(value=kekuleSmiles, style={"display": "none"}, id={'type': 'smi_input', 'index': i}),
    ], style={"display": "inline-block", "border":border, "color":"#ffffff10"}, id={"type":"ImgBlock", "index":i})

    return res


def getHovers(i, params):
    return [dbc.Popover(id={"type":"pred_hover",
                            "param":x.replace(".","___"),  #str + "." はcallback_context.triggeredで失敗するので "___"に置き換える
                            "index":i},
                        target={"type":"pred_param",
                                "param":x.replace(".","___"), #str + "." はcallback_context.triggeredで失敗するので "___"に置き換える
                                "index":i},
                        trigger="hover") for x in params]


def get_pred_panel_td(value, param, i, font_size="", classes=[]):

    if str(value) == "positive" or str(value) == "negative":
        base = html.Td(param,
                   style={**PANEL_STYLE(get_val_color(value, param, classes)), **{"font-size":font_size}},
                   id={"type":"pred_param",
                       "param":param.replace(".","___"), #str + "." はcallback_context.triggeredで失敗するので "___"に置き換える
                       "index":i})
        return [base]
    else:
        base = html.Td("%.1f"%value,
                   style={**PANEL_STYLE(get_val_color(value, param, classes)), **{"font-size":font_size}},
                   id={"type":"pred_param",
                       "param":param.replace(".","___"), #str + "." はcallback_context.triggeredで失敗するので "___"に置き換える
                       "index":i})
        title = html.Td(param, style={"border": "none", "width": "60px", "padding": PADDING_REM, "font-size":font_size})
        return [title, base]


def get_panel(values, params, id, classes=[]):
    if len(params)==0:
        return []

    title = html.Td("", style={"border": "none", "width": "0px", "padding": PADDING_REM})
    top = html.Td("", style={"border": "none", "width": "0px", "padding": PADDING_REM})
    tds = list(itertools.chain.from_iterable([get_pred_panel_td(values[i],x,id, "14px", classes) for i,x in enumerate(params)]))

    res = []

    num_col = 6
    times = int(np.ceil(len(tds)/num_col))

    trs = []
    for i in range(times):
        if i == 0:
            trs.append(html.Tr([""], style={"height": "10px"}))
            trs.append(html.Tr([title] + tds[i*num_col:(i+1)*num_col]))
        else:
            trs.append(html.Tr([top] + tds[i*num_col:(i+1)*num_col]))
    return trs


def getPredPanel(i, smi, pred_val, params=[], use_memo=False, memo="", classes=[]):
    style = {"display":""} if use_memo else {"display":"none"}

    children = get_panel([pred_val[x] for x in params], params, i, classes) + getHovers(i, params)

    temp_panel = html.Div([
        html.Table(children, style={"table-layout": "fixed"}),
        html.Div(dbc.Input(value=memo, id={"type":"memo", "index":i}, style=style))
    ], style={"text-align": "center"})

    return temp_panel


def getFuncPanel(i, smi):
    res = html.Div([
        dbc.Button(text, **FUNCTION_STYLES, id={'type': f'panel_{btn_id}_func', 'index': f"{i}_{smi}"},
                   disabled = False, n_clicks=0)#, className="mb-1")
        for text, btn_id in FUNCTION_BUTTONS
    ])
    return res
