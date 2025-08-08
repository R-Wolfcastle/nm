# constants_units.py
# Physical constants for ice sheet modeling (SI units)

# Unit conversions
S_PER_YEAR = 31_557_600        # 1 year in seconds
T_PER_KG = 1e-3                # Gt in 1 kg

#define a new unit. The spade measures stress in units of T/(m*year^2)
SPADE_PER_PA = T_PER_KG/(S_PER_YEAR**2)



# Fundamental constants (in km, years, Gt)
g = 9.80665 * S_PER_YEAR**2  # m/yr²

# Densities
RHO_I = 917.0  * T_PER_KG     # Density of ice (kg/m^3)
RHO_W = 1028.0 * T_PER_KG     # Density of seawater (kg/m^3)

# Ice rheology (Glen's Flow Law parameters) (converted from Pa⁻³·s⁻¹ to Gt⁻³·km³·yr⁻⁶)
GLEN_N = 3.0                   # Glen's flow law exponent (dimensionless)
A_TEMPERATE = 1.0e-24 *  S_PER_YEAR * SPADE_PER_PA**3  # Rate factor A for temperate ice (Pa^-3 s^-1)
A_COLD = 3.5e-25 * S_PER_YEAR * SPADE_PER_PA**3       # Rate factor A for cold ice (Pa^-3 s^-1)

# Ice viscosity (typical value, can vary widely)
ICE_VISCOSITY = 1.0e13 * SPADE_PER_PA * S_PER_YEAR  # Dynamic viscosity of ice (Pa·s)

# Elastic properties
YOUNGS_MODULUS_ICE = 9.0e9 * SPADE_PER_PA     # Young’s modulus of ice (Pa)
POISSON_RATIO_ICE = 0.3        # Poisson’s ratio for ice (dimensionless)

# Thermal properties
HEAT_CAPACITY_ICE = 2009       # Specific heat capacity (J/kg/K)
THERMAL_CONDUCTIVITY_ICE = 2.1 # W/(m·K)
LATENT_HEAT_FUSION = 3.34e5    # Latent heat of fusion for ice (J/kg)

# Useful reference values
ICE_PRESSURE_MPA_PER_KM = RHO_I * g / 1e6  # ~9.0 MPa per km of ice

# Conversion factors
PA_PER_BAR = 1.0e5             # Pascal per bar

# Misc additions
EPSILON_VISC = 1e-13 / S_PER_YEAR #s^-1 converted to years^-1
