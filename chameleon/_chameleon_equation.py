"""Chameleon equation
"""

from sympy import Symbol, Function, Number, sin, cos, exp
from modulus.pdes import PDES


class ChameleonEquation2D(PDES):
    """
    Chameleon equation 2D

    Parameters
    ==========
    L : float, string
        Potential coefficient.
    """

    name = "ChameleonEquation2D"

    def __init__(self, Lambda=2.4):
        # coordinates
        x = Symbol("x")
        y = Symbol("y")

        # make input variables
        input_variables = {"x": x, "y": y}

        # make u function
        u = Function("u")(*input_variables)

        Lambda = Number(Lambda)

        # set equations
        self.equations = {}
        self.equations["chameleon_equation"] = u.diff(x, 2) + u.diff(y, 2) \
            - sin(x)**2 - sin(y)**2 # + Lambda**5/(1.0e-4 + u**2)
