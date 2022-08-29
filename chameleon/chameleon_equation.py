"""Chameleon equation
Reference: ...
"""

from sympy import Symbol, Function, Number
from modulus.pdes import PDES


class ChameleonEquation(PDES):
    """
    Chameleon equation 1D

    Parameters
    ==========
    c : float, string
        Wave speed coefficient. If a string then the
        wave speed is input into the equation.
    """

    name = "ChameleonEquation"

    def __init__(self, c=1.0):
        # coordinates
        x = Symbol("x")

        # time
        t = Symbol("t")

        # make input variables
        input_variables = {"x": x, "t": t}

        # make u function
        u = Function("u")(*input_variables)

        # wave speed coefficient
        if type(c) is str:
            c = Function(c)(*input_variables)
        elif type(c) in [float, int]:
            c = Number(c)

        # set equations
        self.equations = {}
        self.equations["wave_equation"] = 0.0*u.diff(t, 2) # + u.diff(x, 2) \
            # - u.diff(y, 2) - u.diff(z, 2)
