from sympy import Symbol, pi, sin

import modulus
from modulus.hydra import to_absolute_path, to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.csv_utils.csv_rw import csv_to_dict
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
from modulus.PDES.wave_equation import HelmholtzEquation


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    # make list of nodes to unroll graph on
    wave = HelmholtzEquation(u="u", k=1.0, dim=2)
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = wave.make_nodes() + [wave_net.make_node(name="wave_network", jit=cfg.jit)]

    # add constraints to solver
    # make geometry
    x, y = Symbol("x"), Symbol("y")
    height = 2
    width = 2
    rec = Rectangle((-width / 2, -height / 2), (width / 2, height / 2))

    # make domain
    domain = Domain()

    # walls
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
        outvar={
            "helmholtz": -(
                -((pi) ** 2) * sin(pi * x) * sin(4 * pi * y)
                - ((4 * pi) ** 2) * sin(pi * x) * sin(4 * pi * y)
                + 1 * sin(pi * x) * sin(4 * pi * y)
            )
        },
        batch_size=cfg.batch_size.interior,
        bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
        lambda_weighting={
            "helmholtz": rec.sdf,
        },
    )
    domain.add_constraint(interior, "interior")

    # validation data
    mapping = {"x": "x", "y": "y", "z": "u"}
    openfoam_var = csv_to_dict(to_absolute_path("validation/helmholtz.csv"), mapping)
    openfoam_invar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    }
    openfoam_outvar_numpy = {
        key: value for key, value in openfoam_var.items() if key in ["u"]
    }

    openfoam_validator = PointwiseValidator(
        openfoam_invar_numpy, openfoam_outvar_numpy, nodes, batch_size=1024
    )
    domain.add_validator(openfoam_validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
