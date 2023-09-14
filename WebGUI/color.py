PASTEL_GREEN = "#cfc"
PASTEL_YELLOW = "#ffc"
PASTEL_RED = "#fcc"

def get_val_color(value, param, classes):
    if param in classes:
        if value <= 0.4:
            return PASTEL_GREEN
        if 0.4 < value < 0.6:
            return PASTEL_YELLOW
        if value >= 0.6:
            return PASTEL_RED

def get_atom_color_from_num(n):
    if n == 6: #Carbon
        return "black"
    elif n == 7: #Nitrogen
        return "blue"
    elif n == 8: #Oxygen
        return "red"
    elif n == 9: #Fluoride
        return "rgb(51,204,204)"
    elif n == 17: #Chloride
        return "rgb(0,205,0)"
    elif n == 16: #Sulfur
        return "rgb(204,204,0)"
    elif n == 35: #Bromide
        return "rgb(192,76,25)"
    elif n == 53: #Iodine
        return "rgb(161,30,240)"
    else:
        return "black"
