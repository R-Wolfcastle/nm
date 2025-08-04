# constants.py
# Physical constants for ice sheet modeling (SI units)

# Fundamental constants
G = 9.80665                    # Gravity (m/s^2)
S_PER_YEAR = 31_557_600        # Seconds in a year (365.25 days)

# Densities
RHO_I = 917.0                  # Density of ice (kg/m^3)
RHO_W = 1028.0                 # Density of seawater (kg/m^3)

# Ice rheology (Glen's Flow Law parameters)
GLEN_N = 3.0                   # Glen's flow law exponent (dimensionless)
A_TEMPERATE = 1.0e-24          # Rate factor A for temperate ice (Pa^-3 s^-1)
A_COLD = 3.5e-25               # Rate factor A for cold ice (Pa^-3 s^-1)

# Ice viscosity (typical value, can vary widely)
ICE_VISCOSITY = 1.0e13         # Dynamic viscosity of ice (Pa·s)

# Elastic properties
YOUNGS_MODULUS_ICE = 9.0e9     # Young’s modulus of ice (Pa)
POISSON_RATIO_ICE = 0.3        # Poisson’s ratio for ice (dimensionless)

# Thermal properties
HEAT_CAPACITY_ICE = 2009       # Specific heat capacity (J/kg/K)
THERMAL_CONDUCTIVITY_ICE = 2.1 # W/(m·K)
LATENT_HEAT_FUSION = 3.34e5    # Latent heat of fusion for ice (J/kg)

# Useful reference values
ICE_PRESSURE_MPA_PER_KM = RHO_ICE * G / 1e6  # ~9.0 MPa per km of ice

# Conversion factors
PA_PER_BAR = 1.0e5             # Pascal per bar

# Misc additions
EPSILON_VISC = 1e-13
