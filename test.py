import dsgp4
import torch
from dsgp4.mldsgp4 import mldsgp4
from dsgp4.sgp4init import sgp4, sgp4init

# we load all TLEs:

tles = dsgp4.tle.TLE(
    data=[
        "ISS (ZARYA)",
        "1 25544U 98067A   26038.50283897  .00012054  00000-0  23050-3 0  9996",
        "2 25544  51.6315 221.5822 0011000  74.6214 285.5989 15.48462076551652",
    ]
)

tles = dsgp4.tle.load("example.tle")
# we only extract the first one:
my_tle = tles[0]

# we always have to initialize the TLE before we can use it. If that does not, it can be directly initialized during propagation (with a small performance penalty):
dsgp4.initialize_tle(my_tle)

# I propagate for 1 day:
tsinces = torch.linspace(0, 24 * 60, 10000)
state_teme = dsgp4.propagate(my_tle, tsinces)

dsgp4.plot_orbit(
    state_teme,
    color="lightcoral",
    label=f"SATCAT n°: {my_tle.satellite_catalog_number}",
)

# we first need to prepare the data, the API requires that there are as many TLEs as times. Let us assume we want to
# propagate each of the
tles_ = []
for tle in tles:
    tles_ += [tle] * 10000
tsinces = torch.cat([torch.linspace(0, 24 * 60, 10000)] * len(tles))
# first let's initialize them:
_, tle_batch = dsgp4.initialize_tle(tles_)

# we propagate the batch of 3,000 TLEs for 1 day:
states_teme = dsgp4.propagate_batch(tle_batch, tsinces)

# Let's plot the first orbit:
ax = dsgp4.plot_orbit(
    states_teme[:10000],
    color="lightcoral",
    label=f"SATCAT n°:{tles[0].satellite_catalog_number}",
)
ax = dsgp4.plot_orbit(
    states_teme[10000:20000],
    ax=ax,
    color="darkkhaki",
    label=f"SATCAT n°:{tles[1].satellite_catalog_number}",
)
ax = dsgp4.plot_orbit(
    states_teme[20000:],
    ax=ax,
    color="lightseagreen",
    label=f"SATCAT n°:{tles[2].satellite_catalog_number}",
)
