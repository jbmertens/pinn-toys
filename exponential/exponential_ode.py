from sympy import Symbol, Function, Number, log, exp
from modulus.pdes import PDES


class Exponential(PDES):
    name = "exponential"

    def __init__(self):

        x = Symbol("x")
        input_variables = {"x": x}

        y = Function("y")(*input_variables)

        e = 2.718281828459045
        self.equations = {}
        self.equations["ode_y"] = y - y.diff(x)
