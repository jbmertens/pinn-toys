from sympy import Symbol, Function, Number, log, exp
from modulus.pdes import PDES


class FLRW(PDES):
    name = "FLRW"

    def __init__(self, Ox=(0.65, 0.35, 0.0), h=0.67):

        self.Ox = Ox
        self.h = h

        OL = Number(Ox[0])
        Om = Number(Ox[1])
        Ok = Number(Ox[2])

        lna = Symbol("lna")
        input_variables = {"lna": lna}

        a = Function("a")(*input_variables)
        H = Function("H")(*input_variables)
        phi = Function("phi")(*input_variables)
        psi = Function("psi")(*input_variables)

        e = 2.718281828459045
        self.equations = {}
        self.equations["ode_H"] = H**2 - h**2 *( Om*a**-3.0 + OL )
        self.equations["ode_a"] = a - e**lna # a.diff(lna) # e**lna - a
        self.equations["ode_phi"] = phi - 1.0
        self.equations["ode_psi"] = psi - 1.0
