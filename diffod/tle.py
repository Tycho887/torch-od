import torch
import numpy as np
import datetime

def compute_checksum(line: str) -> int:
    """Computes the TLE line checksum."""
    return sum((int(c) if c.isdigit() else c == '-') for c in line[0:68]) % 10

def format_tle_exp(val: float) -> str:
    """Formats BSTAR and NDDOT terms to the strict 8-character TLE format (e.g., ' 44436-4')."""
    if val == 0.0:
        return " 00000-0"
    
    sign = "-" if val < 0 else " "
    val_abs = abs(val)
    
    # Extract mantissa and exponent (e.g., 4.4436e-05)
    sci_str = f"{val_abs:.4e}"
    mantissa_str, exp_str = sci_str.split('e')
    
    # Remove decimal from mantissa to fit implied decimal format
    mantissa_digits = mantissa_str.replace('.', '')
    
    # Adjust exponent because decimal point was shifted (0.44436 instead of 4.4436)
    tle_exp = int(exp_str) + 1
    
    # Format exponent with sign
    tle_exp_str = f"{tle_exp:d}"
    if tle_exp >= 0:
        tle_exp_str = "+" + tle_exp_str
        
    return f"{sign}{mantissa_digits}{tle_exp_str}".rjust(8, ' ')

def format_tle_ndot(val: float) -> str:
    """Formats NDOT term to 10 characters (e.g., ' .00001878')."""
    sign = "-" if val < 0 else " "
    s = f"{abs(val):.8f}"
    if s.startswith("0."):
        s = s[1:]
    return f"{sign}{s}".ljust(10, ' ')[:10]

# -----------------------------------------------------------------------------
# Single Element Operations
# -----------------------------------------------------------------------------

def tle_decode(tle_lines: list[str]) -> torch.Tensor:
    """Decodes a single TLE string pair into a 1D tensor of 9 parameters."""
    xpdotp = 1440.0 / (2.0 * np.pi)
    
    if len(tle_lines) == 3:
        line1, line2 = tle_lines[1].rstrip(), tle_lines[2].rstrip()
    else:
        line1, line2 = tle_lines[0].rstrip(), tle_lines[1].rstrip()
    
    # --- Line 1 ---
    bstar = float(line1[53] + '.' + line1[54:59]) * (10 ** int(line1[59:61]))
    ndot = float(line1[33:43]) / (xpdotp * 1440.0)
    nddot = (float(line1[44] + '.' + line1[45:50]) * (10 ** int(line1[50:52]))) / (xpdotp * 1440.0 * 1440.0)
    
    # --- Line 2 ---
    inclo = np.deg2rad(float(line2[8:16]))
    nodeo = np.deg2rad(float(line2[17:25]))
    ecco = float('0.' + line2[26:33].replace(' ', '0'))
    argpo = np.deg2rad(float(line2[34:42]))
    mo = np.deg2rad(float(line2[43:51]))
    no_kozai = float(line2[52:63]) / xpdotp
    
    return torch.tensor([bstar, ndot, nddot, ecco, argpo, inclo, mo, no_kozai, nodeo], dtype=torch.float64)

def tle_encode(
    bstar: float, ndot: float, nddot: float, ecco: float, argpo: float,
    inclo: float, mo: float, no_kozai: float, nodeo: float,
    sat_num: int, epoch: datetime.datetime, sat_name: str = None
) -> list[str]:
    """Encodes a single set of parameters back into a TLE string pair."""
    xpdotp = 1440.0 / (2.0 * np.pi)
    
    ndot_val = ndot * xpdotp * 1440.0
    nddot_val = nddot * xpdotp * 1440.0 * 1440.0
    argpo_deg = np.rad2deg(argpo) % 360.0
    inclo_deg = np.rad2deg(inclo) % 360.0
    mo_deg = np.rad2deg(mo) % 360.0
    nodeo_deg = np.rad2deg(nodeo) % 360.0
    no_kozai_val = no_kozai * xpdotp
    
    year = epoch.year
    two_digit_year = year % 100
    start_of_year = datetime.datetime(year - 1, 12, 31)
    epoch_days = (epoch - start_of_year).total_seconds() / 86400.0

    # --- Line 1 (Strict Indexing) ---
    l1 = list("1 " + " " * 67)
    l1[2:7]   = list(str(sat_num).zfill(5)[:5])
    l1[7]     = 'U'
    l1[18:32] = list(f"{two_digit_year:02d}{epoch_days:012.8f}")
    l1[33:43] = list(format_tle_ndot(ndot_val))
    l1[44:52] = list(format_tle_exp(nddot_val))
    l1[53:61] = list(format_tle_exp(bstar))
    l1[62]    = '0'
    l1[64:68] = list(" 999")
    
    l1_str = "".join(l1)[:68]
    l1_str += str(compute_checksum(l1_str))
    
    # --- Line 2 (Strict Indexing) ---
    l2 = list("2 " + " " * 67)
    l2[2:7]   = list(str(sat_num).zfill(5)[:5])
    l2[8:16]  = list(f"{inclo_deg:8.4f}")
    l2[17:25] = list(f"{nodeo_deg:8.4f}")
    
    # Eccentricity has an implied leading decimal (e.g., "0001961")
    l2[26:33] = list(f"{ecco:.7f}"[2:])
    
    l2[34:42] = list(f"{argpo_deg:8.4f}")
    l2[43:51] = list(f"{mo_deg:8.4f}")
    l2[52:63] = list(f"{no_kozai_val:11.8f}")
    l2[63:68] = list("00000")
    
    l2_str = "".join(l2)[:68]
    l2_str += str(compute_checksum(l2_str))
    
    res = [l1_str, l2_str]
    if sat_name:
        res.insert(0, f"0 {sat_name}")
    return res

# -----------------------------------------------------------------------------
# Batch Operations
# -----------------------------------------------------------------------------

def batch_decode(tle_strs: list[list[str]]) -> torch.Tensor:
    return torch.stack([tle_decode(tle) for tle in tle_strs])

def batch_encode(
    bstar: torch.Tensor, ndot: torch.Tensor, nddot: torch.Tensor, 
    ecco: torch.Tensor, argpo: torch.Tensor, inclo: torch.Tensor, 
    mo: torch.Tensor, no_kozai: torch.Tensor, nodeo: torch.Tensor,
    sat_nums: list[int], epochs: list[datetime.datetime], sat_names: list[str]|None = None
) -> list[list[str]]:
    
    N = bstar.shape[0]
    out_tles = []

    if sat_names is None:
        sat_names = [None] * N

    for i in range(N):
        name = sat_names[i] if sat_names else None #name = "spacecraft"
        out_tles.append(tle_encode(
            bstar[i].item(), ndot[i].item(), nddot[i].item(), ecco[i].item(),
            argpo[i].item(), inclo[i].item(), mo[i].item(), no_kozai[i].item(),
            nodeo[i].item(), sat_nums[i], epochs[i], name
        ))
        
    return out_tles