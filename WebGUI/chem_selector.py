import math
import numpy as np
import plotly.express as px

import color
import chem_util

def make_chem_selector(smi):
    atoms, bonds = chem_util.getMolCoordinateInfoFromSmiles2(smi)

    atom_array = np.loadtxt(atoms)[:,:5]

    x_array = atom_array[:,0]
    y_array = atom_array[:,1]
    z_array = atom_array[:,2]

    atom_label_list = [chem_util.ntol(a) for a in atom_array[:,3]]
    atom_color_list = [color.get_atom_color_from_num(a) for a in atom_array[:,3]]

    bond_list = [l.split() for l in bonds]

    x_range = np.max(x_array) - np.min(x_array)
    y_range = np.max(y_array) - np.min(y_array)

    range_lim = max(x_range, y_range)
    font_size = -0.63 * range_lim + 26.4

    grav_x = (np.max(x_array) + np.min(x_array))/2
    grav_y = (np.max(y_array) + np.min(y_array))/2

    fig = px.scatter(x=x_array, y=y_array, text=atom_label_list)

    #レンジ設定用
    fig.add_shape(type="rect",
        x0=grav_x-range_lim/2,
        y0=grav_y-range_lim/2,
        x1=grav_x+range_lim/2,
        y1=grav_y+range_lim/2,
        line=dict(width=0),
    )

    fig.update_traces(
        mode='text',
        textfont=dict(color=atom_color_list),
        textfont_size=font_size
    )
    fig.update_yaxes(visible=False, showticklabels=False)
    fig.update_xaxes(visible=False, showticklabels=False)
    fig.update_layout(margin={'l': 5, 'r': 5, 'b': 5, 't': 5}, plot_bgcolor='rgb(252,252,252)', dragmode='lasso', hovermode=False)

    for i, l in enumerate(bond_list):
        start_idx, end_idx, bond_type, _ = map(int, l)

        x0 = x_array[start_idx-1]
        y0 = y_array[start_idx-1]
        x1 = x_array[end_idx-1]
        y1 = y_array[end_idx-1]

        gx = (x0+x1)/2
        gy = (y0+y1)/2

        scale = 0.55

        x0 = (x0-gx)*scale + gx
        y0 = (y0-gy)*scale + gy

        x1 = (x1-gx)*scale + gx
        y1 = (y1-gy)*scale + gy

        if bond_type == 1:
            fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1, line=dict(width=2))
        elif bond_type == 2:
            slope = (y1-y0)/(x1-x0)

            if slope == 0:
                nor_x_vec = 0
                nor_y_vec = 1
            else:
                x_vec = 1
                y_vec = -1/slope #normal_slope

                nor_x_vec = x_vec / math.sqrt(x_vec**2 + y_vec**2)
                nor_y_vec = y_vec / math.sqrt(x_vec**2 + y_vec**2)

            double_scale = 0.1

            fig.add_shape(type="line",
                          x0=x0+nor_x_vec*double_scale,
                          y0=y0+nor_y_vec*double_scale,
                          x1=x1+nor_x_vec*double_scale,
                          y1=y1+nor_y_vec*double_scale,
                          line=dict(width=2))

            fig.add_shape(type="line",
                          x0=x0-nor_x_vec*double_scale,
                          y0=y0-nor_y_vec*double_scale,
                          x1=x1-nor_x_vec*double_scale,
                          y1=y1-nor_y_vec*double_scale,
                          line=dict(width=2))
        elif bond_type == 3:
            slope = (y1-y0)/(x1-x0)
            normal_slope = -1/slope

            x_vec = 1
            y_vec = normal_slope

            nor_x_vec = x_vec / math.sqrt(x_vec**2 + y_vec**2)
            nor_y_vec = y_vec / math.sqrt(x_vec**2 + y_vec**2)

            triple_scale = 0.2

            fig.add_shape(type="line",
                          x0=x0,
                          y0=y0,
                          x1=x1,
                          y1=y1,
                          line=dict(width=2))

            fig.add_shape(type="line",
                          x0=x0+nor_x_vec*triple_scale,
                          y0=y0+nor_y_vec*triple_scale,
                          x1=x1+nor_x_vec*triple_scale,
                          y1=y1+nor_y_vec*triple_scale,
                          line=dict(width=2))

            fig.add_shape(type="line",
                          x0=x0-nor_x_vec*triple_scale,
                          y0=y0-nor_y_vec*triple_scale,
                          x1=x1-nor_x_vec*triple_scale,
                          y1=y1-nor_y_vec*triple_scale,
                          line=dict(width=2))
    return fig
