# constants.py
# Physical constants for ice sheet modeling (SI units)

# Fundamental constants
#g = 9.80665*(31_557_600**2)    # Gravity (m/a^2)
#It's not actually beneficial to change this. We are justified in keeping the stresses
#in pascals!
g = 9.80665    # Gravity (m/s^2)
S_PER_YEAR = 31_557_600        # Seconds in a year (365.25 days)

# Densities
RHO_I = 917.0                  # Density of ice (kg/m^3)
RHO_W = 1028.0                 # Density of seawater (kg/m^3)

# Ice rheology (Glen's Flow Law parameters)
GLEN_N = 3.0                   # Glen's flow law exponent (dimensionless)
A_TEMPERATE = 1.0e-24*31_557_600          # Rate factor A for temperate ice (Pa^-3 a^-1)
A_COLD = 3.5e-25*31_557_600               # Rate factor A for cold ice (Pa^-3 a^-1)

# Ice viscosity (typical value, can vary widely)
ICE_VISCOSITY = 1.0e13/31_557_600         # Dynamic viscosity of ice (Pa·s)

# Misc additions
EPSILON_VISC = 1e-5
#EPSILON_VISC = 1e-13
