""" This PDE problem was taken from,
"A Physics-Informed Neural Network Framework
for PDEs on 3D Surfaces: Time Independent
Problems" by Zhiwei Fang and Justin Zhan.
"""
from sympy import Symbol, Function

import modulus
from modulus.hydra import to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.continuous.solvers.solver import Solver
from modulus.continuous.domain.domain import Domain
from modulus.geometry.csg.csg_2d import Rectangle
from modulus.continuous.constraints.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.continuous.validator.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
from modulus.pdes import PDES

# define Poisson equation with sympy
class SurfacePoisson(PDES):
    name = "SurfacePoisson"

    def __init__(self):
        # coordinates
        x, y = Symbol("x"), Symbol("y")

        # u
        u = Function("u")(x, y)

        # set equations
        self.equations = {}
        self.equations["poisson_u"] = u.diff(x, 2) + u.diff(y, 2)\
            + 1.0/(u**4 + 1.0e-5) - u

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    # make list of nodes to unroll graph on
    sp = SurfacePoisson()
    poisson_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = sp.make_nodes() + [
        poisson_net.make_node(name="poisson_network", jit=cfg.jit)
    ]

    # add constraints to solver
    # make geometry
    x, y = Symbol("x"), Symbol("y")
    geo = Rectangle( (0, 0), (1, 1) )

    # make domain
    domain = Domain()

    # domain surface
    surface = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"u": 0.0},
        batch_size=cfg.batch_size.surface,
    )
    domain.add_constraint(surface, "surface")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"poisson_u": 0.0},
        batch_size=cfg.batch_size.interior,
        bounds={x: (0, 1), y: (0, 1)},
        lambda_weighting={
            "poisson_u": geo.sdf,
        },
    )
    domain.add_constraint(interior, "interior")

    # validation data
    surface_points = geo.sample_boundary(1024)
    true_solution = {
        "u": surface_points["x"]*0.0 + 1.0
    }
    validator = PointwiseValidator(surface_points, true_solution, nodes, batch_size=128)
    domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
