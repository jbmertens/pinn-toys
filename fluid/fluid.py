""" A gravitating fluid?
"""
from sympy import Symbol, Function, pi, sin
import numpy as np

import modulus
from modulus.hydra import to_yaml, instantiate_arch
from modulus.hydra.config import ModulusConfig
from modulus.continuous.solvers.solver import Solver
from modulus.continuous.domain.domain import Domain
from modulus.geometry.csg.csg_3d import Box
from modulus.continuous.constraints.constraint import (
    PointwiseBoundaryConstraint,
    PointwiseInteriorConstraint,
)
from modulus.continuous.validator.validator import PointwiseValidator
from modulus.key import Key
from modulus.node import Node
from modulus.pdes import PDES

# Custom plot for validation
# define custom class
class CustomValidatorPlotter(ValidatorPlotter):

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables
        x, y, z, t = invar["x"][:,0], invar["y"][:,0],\
            invar["z"][:,0], invar["t"][:,0]
        n = len(x)

        # get and interpolate output variable
        rho_pred, phi_pred, vx_pred, vy_pred, vz_pred =\
            pred_outvar["rho"][:,0], pred_outvar["phi"][:,0],\
            pred_outvar["vx"][:,0], pred_outvar["vy"][:,0], pred_outvar["vz"][:,0],\

        tmin = t[0]
        tmid = t[len(t)//2]
        tmax = t[-1]

        gridify(data, z, zidx, t, tidx)


        fig, (  (ax11, ax12, ax13, ax14, ax15),
                (ax21, ax22, ax23, ax24, ax25),
                (ax31, ax32, ax33, ax34, ax35)
            ) = plt.subplots(3, 5, sharex=True, sharey=True)

        

        # make plot
        f = plt.figure(figsize=(14,4), dpi=100)
        plt.suptitle("Lid driven cavity: PINN vs true solution")
        plt.subplot(1,3,1)
        plt.title("True solution (u)")
        plt.imshow(u_true.T, origin="lower", extent=extent, vmin=-0.2, vmax=1)
        plt.xlabel("x"); plt.ylabel("y")
        plt.colorbar()
        plt.vlines(-0.05, -0.05, 0.05, color="k", lw=10, label="No slip boundary")
        plt.vlines( 0.05, -0.05, 0.05, color="k", lw=10)
        plt.hlines(-0.05, -0.05, 0.05, color="k", lw=10)
        plt.legend(loc="lower right")
        plt.subplot(1,3,2)
        plt.title("PINN solution (u)")
        plt.imshow(u_pred.T, origin="lower", extent=extent, vmin=-0.2, vmax=1)
        plt.xlabel("x"); plt.ylabel("y")
        plt.colorbar()
        plt.subplot(1,3,3)
        plt.title("Difference")
        plt.imshow((u_true-u_pred).T, origin="lower", extent=extent, vmin=-0.2, vmax=1)
        plt.xlabel("x"); plt.ylabel("y")
        plt.colorbar()
        plt.tight_layout()

        return [(f, "custom_plot"),]

    def gridify(data, z, zidx, t, tidx) :
        slice_data = data[(z == z[zidx]) & (t == t[tidx])]
        slice_data = np.split( slice_data, int(np.sqrt(len(slice_data))) )
        return slice_data


# define Fluid equations with sympy
class Fluid(PDES):
    name = "Fluid"

    def __init__(self):
        # coordinates
        x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")

        # fluid "primitives"
        rho = Function("rho")(x, y, z, t) # density
        vx = Function("vx")(x, y, z, t)
        vy = Function("vy")(x, y, z, t)
        vz = Function("vz")(x, y, z, t)
        phi = Function("phi")(x, y, z, t) # metric potential

        # set equations
        self.equations = {}
        self.equations["continuity"] = rho.diff(t) + (rho*vx).diff(x) + (rho*vy).diff(y) + (rho*vz).diff(z)
        self.equations["euler_x"] = vx.diff(t) + vx*vx.diff(x) + vy*vx.diff(y) + vz*vx.diff(z) + phi.diff(x)
        self.equations["euler_y"] = vy.diff(t) + vx*vy.diff(x) + vy*vy.diff(y) + vz*vy.diff(z) + phi.diff(y)
        self.equations["euler_z"] = vz.diff(t) + vx*vz.diff(x) + vy*vz.diff(y) + vz*vz.diff(z) + phi.diff(z)
        self.equations["gravity"] = phi.diff(x, 2) + phi.diff(y, 2) + phi.diff(z, 2) - rho


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    x, y, z, t = Symbol("x"), Symbol("y"), Symbol("z"), Symbol("t")

    # make list of nodes to unroll graph on
    sp = Fluid()
    fluid_net = instantiate_arch(
        input_keys=[Key("x"), Key("y"), Key("z"), Key("t")],
        output_keys=[Key("rho"), Key("vx"), Key("vy"), Key("vz"), Key("phi")],
        periodicity={"x": (0, 1), "y": (0, 1), "z": (0, 1)},
        cfg=cfg.arch.fully_connected,
    )
    nodes = sp.make_nodes() + [
        fluid_net.make_node(name="fluid_network", jit=cfg.jit),
    ]

    # add constraints to solver
    # make geometry
    geo = Box( (0, 0, 0), (1, 1, 1) )
    time_range = {t: (0, 100)}

    # make domain
    domain = Domain()

    # initial condition
    initial = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"rho": 1.0, "vx": 1.0 + 0.2*sin(2*pi*x), "vy": 1.0, "vz": 1.0},
        batch_size=cfg.batch_size.initial,
        bounds={x: (0, 1), y: (0, 1), z: (0, 1)},
        lambda_weighting={"rho": 1.0, "vx": 1.0, "vy": 1.0, "vz": 1.0},
        param_ranges={t: 0.0},
    )
    domain.add_constraint(initial, "initial")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"continuity": 0.0, "euler_x": 0.0, "euler_y": 0.0, "euler_z": 0.0, "gravity": 0.0 },
        batch_size=cfg.batch_size.interior,
        bounds={x: (0, 1), y: (0, 1), z: (0, 1)},
        param_ranges=time_range,
        lambda_weighting={
            "continuity": 1.0,
            "euler_x": 1.0,
            "euler_y": 1.0,
            "euler_z": 1.0,
            "gravity": 1.0,
        },
    )
    domain.add_constraint(interior, "interior")


    deltaT = 0.5
    deltaX = 0.05
    deltaY = 0.05
    deltaZ = 0.05
    x = np.arange(0, 1, deltaX)
    y = np.arange(0, 1, deltaY)
    z = np.arange(0, 1, deltaZ)
    t = np.arange(0, 100, deltaT)
    X, Y, Z, T = np.meshgrid(x, y, z, t)
    X = np.expand_dims(X.flatten(), axis=-1)
    Y = np.expand_dims(Y.flatten(), axis=-1)
    Z = np.expand_dims(Z.flatten(), axis=-1)
    T = np.expand_dims(T.flatten(), axis=-1)
    u = 0.0*np.sin(X)
    invar_numpy = {"x": X, "y": Y, "z": Z, "t": T}
    outvar_numpy = {"phi": u, "rho": u, "vx": u, "vy": u, "vz": u}
    validator = PointwiseValidator( invar_numpy, outvar_numpy, nodes,
        batch_size=128, plotter=CustomValidatorPlotter() )
    domain.add_validator(validator)


    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
