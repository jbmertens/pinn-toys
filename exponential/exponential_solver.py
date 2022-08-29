import numpy as np
from sympy import Symbol, Eq

import modulus
from modulus.hydra import to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.continuous.solvers.solver import Solver
from modulus.continuous.domain.domain import Domain
from modulus.geometry.csg.csg_1d import Point1D
from modulus.continuous.constraints.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseBoundaryConstraint,
)
from modulus.continuous.validator.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node

from exponential_ode import Exponential


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))
    # make list of nodes to unroll graph on
    sm = Exponential()
    sm_net = instantiate_arch(
        input_keys=[Key("x")],
        output_keys=[Key("y")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = sm.make_nodes() + [sm_net.make_node(name="exp_network", jit=cfg.jit)]

    # add constraints to solver
    # make geometry
    geo = Point1D(0)
    x_max =  3.0
    x_min =  1.0
    x_symbol = Symbol("x")
    y = Symbol("y")
    x_range = {x_symbol: (x_min, x_max)}

    # make domain
    domain = Domain()
    
    # initial conditions
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"y": 1.0, "y__x": 1.0},
        batch_size=cfg.batch_size.IC,
        param_ranges={x_symbol: x_min},
    )
    domain.add_constraint(IC, name="IC")
    
    # solve over given time period
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_y": 0.0},
        batch_size=cfg.batch_size.interior,
        param_ranges=x_range,
    )
    domain.add_constraint(interior, "interior")


    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
