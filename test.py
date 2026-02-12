# Assume these modules are imported from the files above
import torch
from dsgp4.newton_method import newton_method
from dsgp4.tle import TLE
from diffod.functional.sgp4 import dsgp4


# ---------------------------------------------------------

# 1. Setup Data
# ---------------------------------------------------------
# Load TLE
TLE_list = [
    "ISS (ZARYA)",
    "1 25544U 98067A   26038.50283897  .00012054  00000-0  23050-3 0  9996",
    "2 25544  51.6315 221.5822 0011000  74.6214 285.5989 15.48462076551652",
]
init_tle = TLE(data=TLE_list)