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

from eb_ode import EinsteinBoltzmann

# Custom plot for validation
# define custom class
class CustomValidatorPlotter(ValidatorPlotter):

    def __call__(self, invar, true_outvar, pred_outvar):
        "Custom plotting function for validator"

        # get input variables
        k, mu, eta = invar["k"][:,0], invar["mu"][:,0], invar["eta"][:,0]
        n = len(k)

        # get and interpolate output variable
        theta_real_pred, theta_imag_pred, d_c_real_pred, d_c_imag_pred, u_c_real_pred,\
            u_c_imag_pred, phi_real_pred, phi_imag_pred, a_pred =\
            pred_outvar["theta_real_pred"][:,0], pred_outvar["theta_imag_pred"][:,0],\
            pred_outvar["d_c_real_pred"][:,0], pred_outvar["d_c_imag_pred"][:,0],\
            pred_outvar["u_c_real_pred"][:,0], pred_outvar["u_c_imag_pred"][:,0],
            pred_outvar["phi_real_pred"][:,0], pred_outvar["phi_imag_pred"][:,0],\
            pred_outvar["a_pred"][:,0]

        etamin = eta[0]
        etamid = eta[len(t)//2]
        etamax = eta[-1]

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


@modulus.main(config_path="conf", config_name="config")
def run(cfg: ModulusConfig) -> None:
    print(to_yaml(cfg))

    k,mu,eta = Symbol("k"), Symbol("mu"), Symbol("eta")

    # make list of nodes to unroll graph on
    sp = EinsteinBoltzmann()
    fluid_net = instantiate_arch(
        input_keys=[Key("k"), Key("mu"), Key("eta")],
        output_keys=[Key("a"), Key("theta_real"), Key("theta_imag"), Key("d_c_real"), Key("d_c_imag"), Key("u_c_real"), Key("u_c_imag"), Key("phi_real"), Key("phi_imag")],
        periodicity={"mu": (0, 1)},
        cfg=cfg.arch.fully_connected,
    )
    nodes = sp.make_nodes() + [
        fluid_net.make_node(name="eb_network", jit=cfg.jit),
    ]

    # add constraints to solver
    # make geometry
    geo = Rectangle( (0.001, 0), (100, 1) )
    eta_range = {eta: (0, 100)}

    # make domain
    domain = Domain()

    # initial condition
    initial = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"a": 0.01, 
                "theta_real": 1.0, 
                "theta_imag": 0.0, 
                "d_c_real": 1.0, 
                "d_c_imag": 0.0, 
                "u_c_real": 1.0, 
                "u_c_imag": 0.0, 
                "phi_real": 1.0, 
                "phi_imag": 0.0},
        batch_size=cfg.batch_size.initial,
        bounds={k: (0.001, 100), mu: (0, 1)},
        lambda_weighting={"a": 1.0, 
                          "theta_real": 1.0,
                          "theta_imag": 1.0, 
                          "d_c_real": 1.0, 
                          "d_c_imag": 1.0, 
                          "u_c_real": 1.0, 
                          "u_c_imag": 1.0, 
                          "phi_real": 1.0, 
                          "phi_imag": 1.0},
        param_ranges={eta: 0.0},
    )
    domain.add_constraint(initial, "initial")

    # interior
    interior = PointwiseInteriorConstraint(
        nodes=nodes,
        geometry=geo,
        outvar={"friedmann_eq": 0.0, 
                "theta_real_eq": 0.0, 
                "theta_imag_eq": 0.0, 
                "d_c_real_eq": 0.0, 
                "d_c_imag_eq": 0.0, 
                "u_c_real_eq": 0.0, 
                "u_c_imag_eq": 0.0, 
                "phi_real_eq": 0.0, 
                "phi_imag_eq": 0.0},
        batch_size=cfg.batch_size.interior,
        bounds={k: (0.001, 100), mu: (0, 1)},
        param_ranges=time_range,
        lambda_weighting={
            "friedmann_eq": 1.0,
            "theta_real_eq": 1.0,
            "theta_imag_eq": 1.0,
            "d_c_real_eq": 1.0,
            "d_c_imag_eq": 1.0,
            "u_c_real_eq": 1.0,
            "u_c_imag_eq": 1.0,
            "phi_real_eq": 1.0,
            "phi_imag_eq": 1.0
        },
    )
    domain.add_constraint(interior, "interior")


    deltaEta = 0.5
    deltaK = 0.05
    deltaMu = 0.05
    k = np.arange(0.001, 100, deltaK)
    mu = np.arange(0, 1, deltaMu)
    eta = np.arange(0, 100, deltaEta)
    K, Mu, Eta = np.meshgrid(k, mu, eta)
    K = np.expand_dims(K.flatten(), axis=-1)
    Mu = np.expand_dims(Mu.flatten(), axis=-1)
    Eta = np.expand_dims(Eta.flatten(), axis=-1)
    u = 0.0*np.sin(X)
    invar_numpy = {"k": K, "mu": Mu, "eta": Eta}
    outvar_numpy = {"a": u, 
                    "theta_real": u, 
                    "theta_imag": u, 
                    "d_c_real": u, 
                    "d_c_imag": u, 
                    "u_c_real": u, 
                    "u_c_imag": u, 
                    "phi_real":u, 
                    "phi_imag"}
    validator = PointwiseValidator( invar_numpy, outvar_numpy, nodes,
        batch_size=128, plotter=CustomValidatorPlotter() )
    domain.add_validator(validator)


    # make solver
    slv = Solver(cfg, domain)

    # start solver
    slv.solve()


if __name__ == "__main__":
    run()
