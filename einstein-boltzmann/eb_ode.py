from sympy import Symbol, Function, Number, log, exp, sqrt
from modulus.pdes import PDES


# define Einstein-Boltzmann equations with sympy
class EinsteinBoltzmann(PDES):
    name = "EinsteinBoltzmann"

    def __init__(self, Ox=(0.67,0.33, 0.0), h=0.67, T_0=2.7):

        self.Ox = Ox
        self.h = h
        self.T_0 = T_0

        OL = Number(Ox[0])
        Om = Number(Ox[1])
        Og = Number(Ox[2])

        # Constants
        c = 1
        hbar = 1
        m_e = 1

        H0 = 100*h
        alpha_fs = 1/137
        sigma_T = 8*np.pi/3 * (alpha_fs*hbar*c/m_e/c**2)**2

        # coordinates
        # eta: conformal time
        # k: wavevector magnitude
        # mu: dot product between photon momentum unit vector and k unit vector
        eta, k, mu = Symbol("eta"), Symbol("k"), Symbol("mu")

        # Background equations
        a = Function("a")(eta, k, mu)

        # Temperature anisotropy
        theta_real = Function("theta_real")(eta, k, mu)
        theta_imag = Function("theta_imag")(eta, k, mu)

        # fluid fields (real and imaginary parts), all defined in k-space
        # CDM density contrast
        d_c_real = Function("d_c_real")(eta, k, mu)
        d_c_imag = Function("d_c_imag")(eta, k, mu)

        # CDM velocity field
        d_c_real = Function("u_c_real")(eta, k, mu)
        d_c_imag = Function("u_c_imag")(eta, k, mu)

        # Metric potentials
        phi_real = Function("phi_real")(eta, k, mu)
        phi_imag = Function("phi_imag")(eta, k, mu)

        # set equations
        self.equations = {}

        # FIXME: Need equation for theta_0_real and theta_0_imag

        # Photon distribution equations
        self.equations["theta_real_eq"] = theta_real.diff("eta") - k*mu*theta_imag + phi_real.diff("eta") + k*mu*phi_imag - 2*sigma_T*(m_e*T_0/2/np.pi)**1.5*exp(-m_e*a/T_0)/sqrt(a)*(theta_0_real - theta_r)
        self.equations["theta_imag_eq"] = theta_imag.diff("eta") + k*mu*theta_real + phi_imag.diff("eta") - k*mu*phi_real - 2*sigma_T*(m_e*T_0/2/np.pi)**1.5*exp(-m_e*a/T_0)/sqrt(a)*(theta_0_imag - theta_imag)

        # CDM fluid equations
        self.equations["d_c_real_eq"] = d_c_real.diff("eta") - k*u_c_imag + 3*phi_real.diff("eta")
        self.equations["d_c_imag_eq"] = d_c_imag.diff("eta") + k*u_c_real + 3*phi_imag.diff("eta")

        self.equations["u_c_real_eq"] = u_c_real.diff("eta") + a.diff("eta") / a *u_c_real + k*phi_imag
        self.equations["u_c_imag_eq"] = u_c_imag.diff("eta") + a.diff("eta") / a *u_c_imag - k*phi_real


        # Metric potential evolution (Einstein equations)
        self.equations["phi_real_eq"] = k**2*phi_real + 3*a.diff("eta")/a*(1 + a.diff("eta")/a)*phi_real.diff("eta") - 12*H0*H0*(Om/a*d_c_real + 4*Og/a**2*theta_0_real)
        self.equations["phi_imag_eq"] = k**2*phi_imag + 3*a.diff("eta")/a*(1 + a.diff("eta")/a)*phi_imag.diff("eta") - 12*H0*H0*(Om/a*d_c_imag + 4*Og/a**2*theta_0_imag)

        # Background/Friedmann equation
        self.equations["friedmann_eq"] = (a.diff("eta"))**2 - H0**2*(Om/a + OL*a**4 + Og)

    def legendre_2(self, mu):

        return (3 mu**2 - 1)/2

    def rho_g(self, a, T_0=2.725):

        # Gives the photon energy density as a function of the scale factor given
        # a value of the CMB temperature today

        return np.pi/15*(T_0/a)**4

    def rho_b(self, a, Omega_m, h):

        # Gives the matter energy density as a function of the scale factor
        # given the fractional matter density today and the hubble parameter

        return 
