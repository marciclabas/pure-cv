import colorsys

def mod_color(i: int, n: int, lightness: float = 0.5, saturation: float = 0.8):
    """Color per every `i mod n`, evenly spaced around the hue circle"""
    hue = i / n
    rgb = colorsys.hls_to_rgb(hue, lightness, saturation)
    return [int(x*255) for x in rgb]