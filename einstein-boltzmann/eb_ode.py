from sympy import Symbol, Function, Number, log, exp, sqrt
from sympy import KroneckerDelta as KD
from modulus.pdes import PDES


# define Einstein-Boltzmann equations with sympy
class EinsteinBoltzmann(PDES):
    name = "EinsteinBoltzmann"

    def __init__(self, max_l, Ox=(0.67, .33, 0.0, 1e-4), h=.67, T_0=2.7):
        
        self.max_l = max_l
        self.Ox = Ox
        self.h = h
        self.T_0 = T_0
        
        OL = Number(Ox[0])
        Oc = Number(Ox[1])
        Ob = Number(Ox[2])
        Og = Number(Ox[3])
        
        # Defining and storing all functions
        self.eta, self.k = Symbol("eta"), Symbol("k")
        eta, k = self.eta, self.k
        
        # Gravity variables
        self.Phi = Function("Phi")(eta, k)
        self.Psi = Function("Psi")(eta, k)
        self.a = Function("a")(eta)
        
        # Fluid fields
        self.d_b = Function("d_b")(eta, k) # Baryon fluid
        self.d_c = Function("d_c")(eta, k)

        self.v_b = Function("v_b")(eta, k) # CDM fluid
        self.v_c = Function("v_c")(eta, k)
        
        
        # Generating list of theta_l functions 
        theta_ls = []
        for l in range(max_l + 1):
            func_name = "Theta_" + str(l)
            func = Function(func_name)(eta, k)
            theta_ls.append(func)
        self.theta_ls = theta_ls

        # Build dictionary of fields
        self.fields = {
            "a": self.a,
            "d_b": self.d_b,
            "d_c": self.d_c,
            "v_b": self.v_b,
            "v_c": self.v_c,
            "Phi": self.Phi,
            "Psi": self.Psi
        }
        for l in range(max_l + 1):
            theta_l_name = "Theta_" + str(l)
            self.fields[theta_l_name] = theta_ls[l]
        
        # Add in theta eqs
        self.theta_l_eqs = self.build_hierarchy()
        
        # Add in Einstein Eqs
        self.einstein_eqs = self.build_gravity()
        
        # Add in fluid equations
        self.fluid_eqs = self.build_fluid()
        
        # Add in scale factor evolution
        self.friedmann_eq = self.build_friedmann()
        
        # Populate equations method
        self.equations = {}
        self.equations.update(self.theta_l_eqs)
        self.equations.update(self.einstein_eqs)
        self.equations.update(self.fluid_eqs)
        self.equations["friedmann_eq"] = self.friedmann_eq
        
        
    def build_hierarchy(self, Gamma=1):

        max_l = self.max_l
        # Given maximum l, builds Boltzmann hierarchy
        eta, k = self.eta, self.k

        # Defining other fields (metric and fluid)
        theta_ls = self.theta_ls
        Phi = self.Phi
        Psi = self.Psi
        v_b = self.v_b
        

        # Building hierarchy of eqs
        # Note that for l=0, theta_ls[l-1] will give the last theta in the list
        # but this isn't a problem because that term will be multiplied by 0
        
        # Also, there is an additional factor of (1 - KD(l+1, max_l)) that serves
        # to remove the l+1 term on the max_l equation so the system will be closed.
        theta_l_eqs = {}
        for l in range(max_l):
            eq = theta_ls[l].diff("eta") + k/(2*l+1)*((1 - KD(l+1, max_l))*(l+1)*theta_ls[l+1] - l*theta_ls[l-1])\
                 - KD(l,0)*Phi.diff("eta") + KD(l,1)*k*Psi/3 \
                 -Gamma*((1-KD(l,0))*theta_ls[l] + KD(1,l)*v_b/3 - KD(l,2)*theta_ls[2]/10)
            
            
            eq_name = "Theta_" + str(l) +"_eq"
            theta_l_eqs[eq_name] = eq

        return theta_l_eqs

    def build_gravity(self):
        
        # Builds scalar metric evolution and constraint equations
        # as well as scale factor evolution (Friedmann equation)
        
        H0 = self.h*100
        Oc = self.Ox[1]
        Ob = self.Ox[2]
        Og = self.Ox[3]
        
        eta, k = self.eta, self.k
        theta_ls = self.theta_ls
        Phi, Psi = self.Phi, self.Psi
        a = self.a
        d_c, d_b = self.d_c, self.d_b

        # Build Einstein Equations
        einstein_evo_eq = k**2*Phi + 3*a.diff("eta")/a*(Phi.diff("eta")-Psi.diff("eta")*a.diff("eta")/a)\
        - 12*H0*H0*(Oc/a*d_c + Ob/a*d_b + 4*Og/a**2*theta_ls[0])
        einstein_constr_eq = k**2*(Phi + Psi) + 12*H0*H0*(Og/a**2*theta_ls[2])

        einstein_eqs = {"einstein_evo_eq": einstein_evo_eq, "einstein_constr_eq": einstein_constr_eq}

        return einstein_eqs
    
    def build_fluid(self, Gamma=1):
        
        # Builds fluid evolution equations for density and velocity fields
        Ob, Og = self.Ox[2], self.Ox[3]
        
        eta, k = self.eta, self.k
        
        d_c = self.d_c
        d_b = self.d_b
        
        v_c = self.v_c
        v_b = self.v_b

        a = self.a
        theta_ls = self.theta_ls
        
        Phi = self.Phi
        Psi = self.Psi
        
        d_c_eq = d_c.diff("eta") - k*v_c + 3*Phi.diff("eta")
        d_b_eq = d_b.diff("eta") - k*v_b + 3*Phi.diff("eta")
        
        v_c_eq = v_c.diff("eta") + a.diff("eta")/a*v_c + k*Psi
        v_b_eq = v_b.diff("eta") + a.diff("eta")/a*v_b + k*Psi - 3/4*Gamma*Ob/Og*a*(v_b + 3*theta_ls[1])
        
        
        fluid_eqs = {"d_c_eq": d_c_eq, "d_b_eq": d_b_eq, "v_c_eq": v_c_eq, "v_b_eq": v_b_eq}
        return fluid_eqs
    
    def build_friedmann(self):
        
        # Builds friedmann equation to govern scale factor evolution
        eta = self.eta
        a = self.a

        H0 = self.h*100
        OL, Oc, Ob, Og = self.Ox
        
        friedmann_eq = (a.diff("eta"))**2 - H0*H0*(Oc/a + Ob/a + OL/a**4 + Og)
        return friedmann_eq
