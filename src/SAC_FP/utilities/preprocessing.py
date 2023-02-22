"""

"""


def normalize(x, scale, inv=False):
    if inv:
        return x*scale
    else:
        return x/scale