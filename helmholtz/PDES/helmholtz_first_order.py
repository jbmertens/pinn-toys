"""First order form of the Helmholtz equation
"""

from sympy import Symbol, Function, Number

from modulus.pdes import PDES


class HelmholtzEquation(PDES):
    name = "HelmholtzEquation"

    def __init__(self, u, k, dim=3):
        """
        Helmholtz equation

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

        # wave speed coefficient
        if type(k) is str:
            k = Function(k)(*input_variables)
        elif type(k) in [float, int]:
            k = Number(k)

        # set equations
        self.equations = {}
        self.equations["helmholtz"] = -(
            k ** 2 * u + dudx.diff(x) + dudy.diff(y) + dudz.diff(z)
        )
        self.equations["compatibility_dudx"] = u.diff(x) - dudx
        self.equations["compatibility_dudy"] = u.diff(y) - dudy
        self.equations["compatibility_dudz"] = u.diff(z) - dudz

        if self.dim < 3:
            self.equations.pop("compatibility_dudz")
        if self.dim == 1:
            self.equations.pop("compatibility_dudy")
