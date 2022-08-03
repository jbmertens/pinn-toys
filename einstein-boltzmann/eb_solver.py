""" A gravitating fluid?
"""
from sympy import Symbol, Function, pi, sin
import numpy as np

import modulus
from modulus.hydra import to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.continuous.solvers.solver import Solver
from modulus.continuous.domain.domain import Domain
from modulus.geometry.csg.csg_1d import Line1D
from modulus.continuous.constraints.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.continuous.validator.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
from modulus.pdes import PDES

from eb_ode import EinsteinBoltzmann



@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    # Define input params
    k,eta = Symbol("k"), Symbol("eta")

    max_l = 4
    Ox = (0.67, 0.33, 0.0, 1e-4)
    h = .67


    # make list of nodes to unroll graph on
    eb = EinsteinBoltzmann(max_l, Ox=Ox, h=h)
    fluid_net = instantiate_arch(
        input_keys=[Key("k"), Key("eta")],
        output_keys=[Key(field_name) for field_name in list(eb.fields.keys())],
        cfg=cfg.arch.fully_connected,
    )
    nodes = eb.make_nodes() + [
        fluid_net.make_node(name="eb_network", jit=cfg.jit),
    ]

    # add constraints to solver
    # make geometry
    geo = Line1D( 0.001, 100 )
    eta_range = {eta: (0, 100)}

    # make domain
    domain = Domain()

    # initial condition
    # FIXME: need better way to set ICS
    ics = {field: 1.0 for field in list(eb.fields.keys())}
    lambdas = {field: 1.0 for field in list(eb.fields.keys())}
    batch_size_init = cfg.batch_size.IC
    print("HEY LOOK HERE", batch_size_init)
    IC = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar=ics,
        batch_size=batch_size_init,
        bounds={k: (0.001, 100)},
        lambda_weighting=lambdas,
        param_ranges={eta: 0.0},
    )
    domain.add_constraint(IC, "IC")

    # interior
    intr = {eq: 0.0 for eq in list(eb.equations.keys())}
    lambdas_eq = {eq: 1.0 for eq in list(eb.equations.keys())}
    batch_size_intr = cfg.batch_size.interior
    print(batch_size_intr)
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar=intr,
        batch_size=batch_size_intr,
        bounds={k: (0.001, 100)},
        param_ranges=time_range,
        lambda_weighting=lambdas_eq,
    )
    domain.add_constraint(interior, "interior")


    # FIXME: placeholder validator
    deltaEta = 0.5
    deltaK = 0.05
    k = np.arange(0.001, 100, deltaK)
    eta = np.arange(0, 100, deltaEta)
    K, Eta = np.meshgrid(k, eta)
    K = np.expand_dims(K.flatten(), axis=-1)
    Eta = np.expand_dims(Eta.flatten(), axis=-1)
    u = 0.0*np.sin(X)
    invar_numpy = {"k": K, "eta": Eta}
    outvar_numpy = {field_name:u for field_name in list(eb.fields.keys())}
    validator = PointwiseValidator( invar_numpy, outvar_numpy, nodes,
        batch_size=128, plotter=CustomValidatorPlotter() )
    domain.add_validator(validator)


    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
