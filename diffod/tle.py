import torch
import numpy as np
import datetime

def compute_checksum(line: str) -> int:
    """Computes the TLE line checksum."""
    return sum((int(c) if c.isdigit() else c == '-') for c in line[0:68]) % 10

# -----------------------------------------------------------------------------
# Single Element Operations
# -----------------------------------------------------------------------------

def tle_decode(tle_lines: list[str]) -> torch.Tensor:
    """Decodes a single TLE string pair into a 1D tensor of 9 parameters."""
    xpdotp = 1440.0 / (2.0 * np.pi)
    
    # Handle optional Line 0 (Name)
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
    sat_num: int, epoch: datetime.datetime, sat_name: str|None = None
) -> list[str]:
    """Encodes a single set of parameters back into a TLE string pair."""
    xpdotp = 1440.0 / (2.0 * np.pi)
    
    # Convert to TLE native units
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
    
    # --- Line 1 ---
    l1 = ['1 ']
    l1.append(str(sat_num).zfill(5)[:5])
    l1.append('U ')
    l1.append('        ')  # 8 spaces for empty international designator
    l1.append(f"{two_digit_year:02d}{epoch_days:012.8f} ")
    l1.append('{0: 8.8f}'.format(ndot_val).replace('0', '', 1) + ' ')
    l1.append('{0: 4.4e}'.format(nddot_val * 10).replace(".", '').replace('e+00', '-0').replace('e-0', '-').replace('e+0', '+') + ' ')
    l1.append('{0: 4.4e}'.format(bstar * 10).replace('.', '').replace('e+0', '+').replace('e-0', '-') + ' ')
    l1.append('0 ')   # Ephemeris type + space
    l1.append(' 999') # Element number padded to 4 chars
    
    # Force exact 68 character length before checksum
    l1_str = ''.join(l1).ljust(68)
    l1_str += str(compute_checksum(l1_str))
    
    # --- Line 2 ---
    l2 = ['2 ']
    l2.append(str(sat_num).zfill(5)[:5] + ' ')
    l2.append('{0:8.4f}'.format(inclo_deg).rjust(8, ' ') + ' ')
    l2.append('{0:8.4f}'.format(nodeo_deg).rjust(8, ' ') + ' ')
    l2.append(str(round(ecco * 1e7)).rjust(7, '0')[:7] + ' ')
    l2.append('{0:8.4f}'.format(argpo_deg).rjust(8, ' ') + ' ')
    l2.append('{0:8.4f}'.format(mo_deg).rjust(8, ' ') + ' ')
    l2.append('{0:11.8f}'.format(no_kozai_val).rjust(11, ' '))
    l2.append('00000') # Revolution number at epoch (defaulted to 0 to save tensor space)
    
    # Force exact 68 character length before checksum
    l2_str = ''.join(l2).ljust(68)
    l2_str += str(compute_checksum(l2_str))
    
    res = [l1_str, l2_str]
    if sat_name:
        res.insert(0, f"0 {sat_name}")
    return res

# -----------------------------------------------------------------------------
# Batch Operations
# -----------------------------------------------------------------------------

def batch_decode(tle_strs: list[list[str]]) -> torch.Tensor:
    """
    Decodes a list of TLE string pairs into a batched tensor.
    Returns: torch.Tensor of shape (N, 9)
    """
    return torch.stack([tle_decode(tle) for tle in tle_strs])

def batch_encode(
    bstar: torch.Tensor, ndot: torch.Tensor, nddot: torch.Tensor, 
    ecco: torch.Tensor, argpo: torch.Tensor, inclo: torch.Tensor, 
    mo: torch.Tensor, no_kozai: torch.Tensor, nodeo: torch.Tensor,
    sat_nums: list[int], epochs: list[datetime.datetime], sat_names: list[str]|None = None
) -> list[list[str]]:
    """
    Encodes batched parameter tensors into a list of TLE string arrays.
    """
    N = bstar.shape[0]
    out_tles = []
    
    for i in range(N):
        name = sat_names[i] if sat_names else None
        
        out_tles.append(tle_encode(
            bstar[i].item(), ndot[i].item(), nddot[i].item(), ecco[i].item(),
            argpo[i].item(), inclo[i].item(), mo[i].item(), no_kozai[i].item(),
            nodeo[i].item(), sat_nums[i], epochs[i], name
        ))
        
    return out_tles