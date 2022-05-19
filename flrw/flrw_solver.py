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

from flrw_ode import FLRW


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))
    # make list of nodes to unroll graph on
    sm = FLRW(Ox=(0.7, 0.3, 0.0), h=0.67)
    sm_net = instantiate_arch(
        input_keys=[Key("lna")],
        output_keys=[Key("a"), Key("H"), Key("phi"), Key("psi")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = sm.make_nodes() + [sm_net.make_node(name="flrw_network", jit=cfg.jit)]

    # add constraints to solver
    # make geometry
    geo = Point1D(0)
    lna_max =  0.0
    lna_min = -1.5
    lna_symbol = Symbol("lna")
    lna_range = {lna_symbol: (lna_min, lna_max)}

    # make domain
    domain = Domain()
    
    # initial conditions
    IC = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"a": 1.0, "H": 0.67,
                "phi": 1.0, "psi": 1.0},
        batch_size=cfg.batch_size.IC,
        lambda_weighting={
            "a": 1.0,
            "H": 1.0,
            "phi": 1.0,
            "psi": 1.0,
        },
        param_ranges={lna_symbol: lna_max},
    )
    domain.add_constraint(IC, name="IC")
    
    # solve over given time period
    interior = PointwiseBoundaryConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"ode_a": 0.0, "ode_H": 0.0,
                "ode_phi": 0.0, "ode_psi": 0.0},
        batch_size=cfg.batch_size.interior,
        param_ranges=lna_range,
    )
    domain.add_constraint(interior, "interior")

    # # add validation data
    # n_a = 200
    # lna = np.linspace(lna_min, lna_max, n_a)
    # lna = np.expand_dims(lna, axis=-1)
    # invar_numpy = {"lna": lna}
    # outvar_numpy = {
    #     "H": 1.0*lna,
    #     "a": 1.0*lna,
    #     "phi": 1.0*lna,
    #     "psi": 1.0*lna,
    # }
    # validator = PointwiseValidator(invar_numpy, outvar_numpy, nodes, batch_size=1024)
    # domain.add_validator(validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
