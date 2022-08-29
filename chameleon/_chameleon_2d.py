import numpy as np
from sympy import Symbol

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
from modulus.continuous.inferencer.inferencer import PointwiseInferencer
from modulus.key import Key
from modulus.node import Node
from chameleon_equation import ChameleonEquation2D


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))
    print(type(cfg.optimizer))
    print(cfg.optimizer)

    # make list of nodes to unroll graph on
    ce = ChameleonEquation2D(Lambda=2.4)
    chameleon_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = ce.make_nodes() + [chameleon_net.make_node(name="chameleon_network", jit=cfg.jit)]

    # add constraints to solver
    # make geometry
    x, y = Symbol("x"), Symbol("y")
    L = float(np.pi)
    rec = Rectangle((0, L), (0, L))

    # make domain
    domain = Domain()

    # boundary conditions
    wall = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"u": 0},
        batch_size=cfg.batch_size.wall,
    )
    domain.add_constraint(wall, "wall")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=rec,
        outvar={"chameleon_equation": 0},
        batch_size=cfg.batch_size.interior,
        bounds={x: (0, L), y: (0, L)},
        lambda_weighting={"chameleon_equation": rec.sdf,},
    )
    domain.add_constraint(interior, "interior")

    # add validation data
    # deltaT = 0.01
    # deltaX = 0.01
    # x = np.arange(0, L, deltaX)
    # t = np.arange(0, 2 * L, deltaT)
    # X, T = np.meshgrid(x, t)
    # X = np.expand_dims(X.flatten(), axis=-1)
    # T = np.expand_dims(T.flatten(), axis=-1)
    # u = np.sin(X) * (np.cos(T) + np.sin(T))
    # invar_numpy = {"x": X, "t": T}
    # outvar_numpy = {"u": u}
    # validator = PointwiseValidator(invar_numpy, outvar_numpy, nodes, batch_size=128)
    # domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
