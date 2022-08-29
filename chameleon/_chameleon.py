import torch
from sympy import Symbol
from typing import Dict

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
from modulus.geometry.csg.adf import ADF
from PDES.chameleon_first_order import ChameleonEquation

class HardBC(ADF):
    def __init__(self):
        super().__init__()

        # domain measures
        self.domain_height: float = 2.0
        self.domain_width: float = 2.0

        # boundary conditions (bottom, right, top, left)
        self.g: List[float] = [0.0, 0.0, 0.0, 0.0]

        # parameters
        self.eps: float = 1e-9
        self.mu: float = 2.0
        self.m: float = 2.0

    def forward(self, invar: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

        """
        Forms the solution anstaz for the Helmholtz example
        """

        outvar = {}
        x, y = invar["x"], invar["y"]
        omega_0 = ADF.line_segment_adf(
            (x, y),
            (-self.domain_width / 2, -self.domain_height / 2),
            (self.domain_width / 2, -self.domain_height / 2),
        )
        omega_1 = ADF.line_segment_adf(
            (x, y),
            (self.domain_width / 2, -self.domain_height / 2),
            (self.domain_width / 2, self.domain_height / 2),
        )
        omega_2 = ADF.line_segment_adf(
            (x, y),
            (self.domain_width / 2, self.domain_height / 2),
            (-self.domain_width / 2, self.domain_height / 2),
        )
        omega_3 = ADF.line_segment_adf(
            (x, y),
            (-self.domain_width / 2, self.domain_height / 2),
            (-self.domain_width / 2, -self.domain_height / 2),
        )
        omega_E_u = ADF.r_equivalence([omega_0, omega_1, omega_2, omega_3], self.m)

        bases = [
            omega_0 ** self.mu,
            omega_1 ** self.mu,
            omega_2 ** self.mu,
            omega_3 ** self.mu,
        ]
        w = [
            ADF.transfinite_interpolation(bases, idx, self.eps)
            for idx in range(len(self.g))
        ]
        g = w[0] * self.g[0] + w[1] * self.g[1] + w[2] * self.g[2] + w[3] * self.g[3]
        outvar["u"] = g + omega_E_u * invar["u_star"]
        return outvar

@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    # make list of nodes to unroll graph on
    wave = ChameleonEquation(u="u", dim=2)
    hard_bc = HardBC()
    wave_net = instantiate_arch(
        input_keys=[Key("x"), Key("y")],
        output_keys=[Key("u")],
        cfg=cfg.arch.fully_connected,
    )
    nodes = (
        wave.make_nodes()
        + [Node(inputs=["x", "y"], outputs=["u"], evaluate=hard_bc)]
        + [wave_net.make_node(name="wave_network", jit=cfg.jit)]
    )

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
            "chameleon": 0.0,
        },
        batch_size=cfg.batch_size.interior,
        bounds={x: (-width / 2, width / 2), y: (-height / 2, height / 2)},
        lambda_weighting={
            "chameleon": rec.sdf,
        },
    )
    domain.add_constraint(interior, "interior")

    # # validation data
    # mapping = {"x": "x", "y": "y", "z": "u"}
    # openfoam_var = csv_to_dict(to_absolute_path("validation/helmholtz.csv"), mapping)
    # openfoam_invar_numpy = {
    #     key: value for key, value in openfoam_var.items() if key in ["x", "y"]
    # }
    # openfoam_outvar_numpy = {
    #     key: value for key, value in openfoam_var.items() if key in ["u"]
    # }
    # openfoam_validator = PointwiseValidator(
    #     openfoam_invar_numpy, openfoam_outvar_numpy, nodes, batch_size=1024
    # )
    # domain.add_validator(openfoam_validator)

    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
