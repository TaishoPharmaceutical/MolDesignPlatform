#!/usr/bin/env python
# coding: utf-8

import os
import dash
import dash_bootstrap_components as dbc
from dash import callback_context, dcc, html, dash_table, Input, Output, State, ALL
import json

from app import app
import callbacks
import panel_modal
from utils import get_downloader_menu, get_param
from constants import MAIN_URL, config, root_path, webgui_path

form = html.Div(
    [
        dbc.Form(
            dbc.Row([
                    dbc.Label("SMILES", width="auto"),
                    dbc.Col(dbc.Textarea(value="", id="smiles", placeholder="SMILES", style={"height": "75px"})),
                    dbc.Col(dbc.Button("Submit", color="primary", id="submit"), width="auto"),
                ], className="g-2"),
            style={"width": "50%"}),
    ],
    style={"position": "relative", "left": "25%"}, id = "smiles_input_form"
)

menu1 = dbc.DropdownMenu(
    label="File",
    children=[
        dbc.DropdownMenuItem("Save", id="save_button", n_clicks=0),
        dcc.Download(id="save_downloader"),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Selected compounds", header=True),
        dbc.DropdownMenuItem("Unselect", id="whole_clear", style={"display":"inline-block"}),
        dbc.DropdownMenuItem("Open in a new tab", id="new_tab", target="_blank", style={"display":"inline-block"}),
        get_downloader_menu(999, display="inline-block", size=""),
    ],
    color="dodgerblue",
)


menu2 = dbc.DropdownMenu(
    label="Input",
    children=[
        dbc.DropdownMenuItem("Unshow", id="menu_smiles_input", n_clicks=0),
    ],
    color="dodgerblue",
)


menu3 = dbc.DropdownMenu(
    label="Settings",
    children=[
        dbc.DropdownMenuItem("Predictions", id="pred_others", n_clicks=0),
        dbc.Popover(
            [
                dcc.Checklist(
                    options=[
                        {'label': x, 'value': x} for x in get_param()["all"]
                    ],
                    value=get_param()["all"],
                    id="pred_others_check",
                    style={"font-size":"medium"},
                    labelStyle={'display': 'block'}
                ),
                dcc.Store(id="pred_others_store"),
                dcc.Store(data=get_param(), id="pred_param"),
                dbc.Button("OK", id="pred_others_ok",outline=True,
                           color="primary", className="me-1"),
                dbc.Button("CANCEL", id="pred_others_cancel",outline=True,
                           color="primary", className="me-1"),
            ],
            target= "pred_others",
            trigger="hover",
            hide_arrow=True,
            id = "pred_others_pop",style={"border":"groove"}
        ),
        dbc.DropdownMenuItem(divider=True),
        dcc.Checklist(
            options=[
                {'label': "AI-attention", 'value': "attention"}
            ],
            value=[],
            id="attention_check",
            style={"font-size":"medium", "height":"100%", "text-align":"middle", "margin-left":"30px"},
            labelStyle={'display': 'inline'}
        ),
        dcc.Checklist(
            options=[
                {'label': "Use memo", 'value': "use_memo"}
            ],
            value=[],
            id="use_memo",
            style={"font-size":"medium", "height":"100%", "text-align":"middle", "margin-left":"30px"},
            labelStyle={'display': 'inline'}
        ),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Display", id = "display_style"),
        dbc.Popover(
            dbc.Tabs(
                [
                    dbc.Tab(label="List View", tab_id="list_view"),
                    dbc.Tab(label="Table View", tab_id="table_view"),
                ],
                id="view_tabs",
                active_tab="list_view",
            ),
            target="display_style",
            trigger="hover"
        ),
        dbc.DropdownMenuItem(divider=True),
        dbc.DropdownMenuItem("Show columns", header=True),
        dcc.Checklist(
            options=[],
            value=[],
            id = "result_dropdown",
            style = {"margin-left":"30px"}
        )
    ],
    color="dodgerblue",
)

menu4 = dbc.DropdownMenu(
    label="Help",
    children=[
        dbc.DropdownMenuItem("Manual",
                             target="_blank",
                             href="",
                             style={"display":"inline-block"}),
        dbc.DropdownMenuItem("Webpage",
                     href="",
                     target="_blank"),
    ],
    color="dodgerblue",
)


header = dbc.Row(
    html.Div([
        dcc.Location(id="url", refresh=False),
        dcc.Input(value="", id="cashe_smiles", style={"display":"none"}),
        html.Tr([
            html.Td(style={"width":"20px"}),
            html.Td(
                html.A(html.Img(src=f"assets/logo.svg",
                                 style={"width":"100px"}),
                       href = MAIN_URL),
                style={"height":"60px", "vertical-align":"middle"}
            ),
            html.Td(style={"width":"20px"}),
            html.Td(menu1,style={"height":"60px", "vertical-align":"middle"}),
            html.Td(menu2,style={"height":"60px", "vertical-align":"middle"}),
            html.Td(menu3,style={"height":"60px", "vertical-align":"middle"}),
            #html.Td(menu4,style={"height":"60px", "vertical-align":"middle"}),
            html.Td(html.H2("MolDesignPlatform", style={"color":"white"}),style={"position": "relative", "left": "15%"}),
        ])
    ], style={"display":"flex", "background-color":"dodgerblue", "width":"100%"})
)

def index():
    return html.Div([
        header,
        html.Br(),
        form,
        html.Div("", id="func"),

        # AIで作成したSmilesの一時保存場所
        html.Div(
            [html.Div("", id={"type":"result", "index":i}, style={"display":"none"}) for i in range(4)],
            id = "results"
        ),

        dcc.Store(id = "user_id"),

        dbc.Container(
            [
                html.Br(),
                dbc.Table(
                    [
                        dbc.Row(children = [],id = "output", style={"display":"flex", "flex-wrap":"nowrap"}),
                    ],
                    id = "result_table"
                )
            ],
            fluid=True,
            id = "result_container",
        ),

        panel_modal.react_modal,
        panel_modal.opt1_modal,
        panel_modal.linker_modal,
        panel_modal.save_modal,
        panel_modal.loading_modal
    ], id="index_html")

app.layout = index()

if __name__ == '__main__':
    os.makedirs(f"{webgui_path}/temp", exist_ok=True)

    if config["ssl"]["key"]=="":
        app.run_server(host=config["ip"]["WebGUI"],
                       port=config["port"]["WebGUI"], debug=True)
    else:
        app.run_server(host=config["ip"]["WebGUI"],
                       port=config["port"]["WebGUI"], debug=True,
                       ssl_context=(config["ssl"]["crt"], config["ssl"]["key"]))
