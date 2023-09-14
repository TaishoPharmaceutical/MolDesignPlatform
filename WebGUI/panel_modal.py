from dash import dcc, html
import dash_bootstrap_components as dbc

from app import app
#import chem_selector

from constants import SCAFFOLD_PRESET, FLAG_LINK_PRESET, GROWING_PRESET, STRUCTURE_FILTER

REACTION_TYPE = {
    "condensation-with-amine": "Condensation (with amine)",
    "condensation-with-carboxylic": "Condensation (with carboxylic acid)",
    "suzuki-with-arylhalide":"Suzuki coupling (with arylhalide)",
    "suzuki-with-boron":"Suzuki coupling (with boronic acid)",
    "reductive-amination-with-amine":"Reductive amination (with amine)",
    "reductive-amination-with-aldehyde-or-ketone":"Reductive amination (with aldehyde or ketone)",
    "buchwald-amination-with-amine":"Buchwald amination (with amine)",
    "buchwald-amination-with-arylhalide":"Buchwald amination (with arylhalide)"}

react_modal = html.Div([
    dbc.Modal([
        dbc.ModalHeader("Reaction with the reagents"),
        dbc.ModalBody([
            dbc.RadioItems(
                id="react-radio",
                options=[ {"label": REACTION_TYPE[x], "value": x} for x in REACTION_TYPE.keys()],
                inline=True,
                value=list(REACTION_TYPE.keys())[0],
            ),
            html.Br(),
            dbc.Label("Reagent conditions", lg=12),
            dbc.Row([
                dbc.Label(f"Mol Weight:", width=3),
                dbc.Col(dbc.Input(id=f"MolWt_min"), width=2),
                dbc.Label("～", width=0.5),
                dbc.Col(dbc.Input(id=f"MolWt_max"), width=2),
                dbc.Label(" ", width="auto"),
            ], className="mb-2"),
            dbc.Row([
                dbc.Label(f"Num of aromatic rings:", width=3),
                dbc.Col(dbc.Input(id=f"ArNum_min"), width=2),
                dbc.Label("～", width=0.5),
                dbc.Col(dbc.Input(id=f"ArNum_max"), width=2),
                dbc.Label(" ", width="auto"),
            ], className="mb-2"),

            html.Br(),
            dbc.Progress(id="progress", striped=True, style={"visibility": "hidden"}),
            dcc.Interval(id="interval", interval=100, disabled=True),
            html.Div("0%", id="progress_text", style={"visibility": "hidden"}),
        ]),
        dbc.ModalFooter([
            html.Div(id="react-button"),
        ]),
    ], id="react_modal",),
])


def get_filter_dropdown(function="linking"):
    res = html.Div([
        html.Pre("Structure Filter Level: "),
        dcc.Dropdown(
            options = [
                {'label': x, 'value': x} for x in STRUCTURE_FILTER
            ],
            value = "basic",
            id = {"type":"filter", "index":function},
        )
    ], style={"display":"inline"})

    return res

# linker_sel = chem_selector.Selector(app, "", "linker-sel", "linker_result")

def get_linking_form(func = "scaffold"):
    if func == "scaffold":
        d = SCAFFOLD_PRESET
    elif func == "linking":
        d = FLAG_LINK_PRESET
    else:
        d = GROWING_PRESET

    radio = dcc.RadioItems(
        options=[
            {'label': 'ScaffoldHopping', 'value': 'sh'},
            {'label': "FragmentLinking", 'value': 'fl'},
            {'label': "FragmentGrowing", 'value': 'fg'}
        ],
        value='sh',
        style={"display":"inline", "font-size":"small"},
        id ="linker-radio"
    )

    return html.Div([radio])


def get_linker_button_form():
    return  [
            dbc.Button(
                ["Scaffold",html.Br(),"Hopping"],
                id={'type': 'panel_linker_func', 'index': f"hopping"},
                color="primary",
                className="me-md-2",
                style={"width":"125px"},
                outline=True,
                n_clicks=0,
            ),
            "   ",
            dbc.Button(
                ["Fragment",html.Br(),"Growing"],
                id={'type': 'panel_linker_func', 'index': f"growing"},
                color="primary",
                className="me-md-2",
                style={"width":"125px"},
                outline=True,
                n_clicks=0,
            ),
    ]



linker_modal = html.Div([
    dbc.Modal([
        dbc.ModalHeader("Subst-gen"),
        dbc.ModalBody([
            dcc.RadioItems(
                options=[
                    {'label': 'FragmentVAE', 'value': 'fragment_vae'},
                ],
                value='fragment_vae',
                style={"display":"inline", "width":"100vw"},
                id ="choose_ai_model"
            ),
            html.H6("Please select the substructure you want to convert.", style={"color":"royalblue"}),
            dbc.Container([
                #linker_sel.view(),
                html.Div(id="linker-sel2"),
                dbc.Input(id="linker_result", style={"display": "none"}),
            ], style={"text-align": "center"}),
            html.Br()
        ]),
        dbc.ModalBody([get_filter_dropdown(function="linking")]),
        dbc.ModalBody(
            html.Div(get_linker_button_form(),
                     id="linker-button", style={"display":"block"}),
        ),
        dbc.ModalFooter([
            dbc.Button("Close", id={'type': 'panel_linker_func', 'index': "close"}, n_clicks=0),
        ]),
    ],id="linker_modal", is_open=False),
])

opt_param_list = [
    {
        "Sol.": "Solubility(ug/mL)",
        "LogD": "LogD",
        "HLMS": "HLMS"
    },
    {
        "CYP1A2": "CYP1A2",
        "CYP2C19": "CYP2C19",
        "CYP2C9": "CYP2C9",
        "CYP2D6": "CYP2D6",
        "CYP3A4": "CYP3A4"
    },
    {
        "Cyto10": "Cyto_10uM",
        "Cyto50": "Cyto_50uM",
        "hERG1": "hERG_1uM",
        "hERG10": "hERG_10uM",
        "Pgp": "Pgp",
        "Ames": "Ames"
    }
]

opt1_modal = html.Div([
    dbc.Modal([
        dbc.ModalHeader("Peripheral compound generator"),
        dbc.ModalBody([get_filter_dropdown(function="opt")]),
        dbc.ModalFooter([
            html.Div(id="opt-button"),
        ]),
    ],id="opt1_modal",),
])


dock_modal = html.Div([
    dbc.Modal([
        dbc.ModalHeader("Docking simulation"),
        dbc.ModalBody(["Docking simulation"]),
        dbc.ModalFooter([
            dcc.Link(dbc.Button("Submit", id={'type': 'panel_dock_func', 'index': "submit"}), href="/dock?kokoni_smiles_toka", target="_blank"),
            dbc.Button("Close", id={'type': 'panel_dock_func', 'index': "close"}),
        ]),
    ], id="dock_modal", ),
])


save_modal = dbc.Modal([
    dbc.ModalHeader("Save"),
    dbc.ModalFooter([
        dcc.Store(id="save_path"),
        dbc.Button("Save", id="save_over_button", n_clicks=0),
        dbc.Button("Save as", id="saveas_button", n_clicks=0),
    ]),
],id="save_modal", is_open=False)


loading_modal = dbc.Modal([
    dbc.ModalBody([
        dbc.Button(
            [dbc.Spinner(size="md"), " Updating..."],
            id = "loading_button",
            color="primary",
            disabled=True,
            n_clicks=0,
            style={"width":"100%"}
        ),
    ]),
],id="loading_modal", is_open=False)
