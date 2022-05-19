"""Chameleon equation
"""

from sympy import Symbol, Function, Number
from sympy import Symbol, pi, sin

from modulus.pdes import PDES


class ChameleonEquation(PDES):
    name = "ChameleonEquation"

    def __init__(self, u, dim=2):
        """
        Chameleon equation solution
        (based on Helmholtz equation solution)

        Parameters
        ==========
        u : str
            The dependent variable.
        k : float, Sympy Symbol/Expr, str
            Wave number. If `k` is a str then it is
            converted to Sympy Function of form 'k(x,y,z,t)'.
            If 'k' is a Sympy Symbol or Expression then this
            is substituted into the equation.
        dim : int
            Dimension of the wave equation (1, 2, or 3). Default is 2.
        """

        # set params
        self.u = u
        self.dim = dim

        # coordinates
        x, y, z = Symbol("x"), Symbol("y"), Symbol("z")

        # make input variables
        input_variables = {"x": x, "y": y, "z": z}
        if self.dim == 1:
            input_variables.pop("y")
            input_variables.pop("z")
        elif self.dim == 2:
            input_variables.pop("z")

        # Scalar function
        assert type(u) == str, "u needs to be string"
        u = Function(u)(*input_variables)
        dudx = Function("dudx")(*input_variables)
        if self.dim == 2:
            dudy = Function("dudy")(*input_variables)
        else:
            dudy = Number(0)
        if self.dim == 3:
            dudz = Function("dudz")(*input_variables)
        else:
            dudz = Number(0)

        # set equations
        LL = 2.4
        self.equations = {}
        self.equations["chameleon"] = -(
            dudx.diff(x) + dudy.diff(y) + dudz.diff(z)\
            + LL**5 / u**2 \
            + -(
                -((pi) ** 2) * sin(pi * x) * sin(4 * pi * y)
                - ((4 * pi) ** 2) * sin(pi * x) * sin(4 * pi * y)
                + 1 * sin(pi * x) * sin(4 * pi * y)
            )**2
        )
        self.equations["compatibility_dudx"] = u.diff(x) - dudx
        self.equations["compatibility_dudy"] = u.diff(y) - dudy
        self.equations["compatibility_dudz"] = u.diff(z) - dudz

        if self.dim < 3:
            self.equations.pop("compatibility_dudz")
        if self.dim == 1:
            self.equations.pop("compatibility_dudy")
