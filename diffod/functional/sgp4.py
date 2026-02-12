import torch
import numpy as np
from typing import NamedTuple

# You will need to ensure initl is functional (tensor in -> tensor out)
from .initl import initl 

class GravConsts(NamedTuple):
    """Gravitational constants (e.g. WGS-72, WGS-84)."""
    tumin: float
    mu: float
    radiusearthkm: float
    xke: float
    j2: float
    j3: float
    j4: float
    j3oj2: float

def sgp4_propagate(
    consts: GravConsts,
    tsince: torch.Tensor,
    bstar: torch.Tensor,
    ndot: torch.Tensor,
    nddot: torch.Tensor,
    ecco: torch.Tensor,
    argpo: torch.Tensor,
    inclo: torch.Tensor,
    mo: torch.Tensor,
    no_kozai: torch.Tensor,
    nodeo: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A functional, JIT-compilable, vmap-friendly SGP4 propagator.
    
    Args:
        consts: Gravitational constants.
        tsince: Time in minutes since epoch.
        ...TLE parameters as tensors...
        
    Returns:
        r: Position vectors (km) [Batch, 3]
        v: Velocity vectors (km/s) [Batch, 3]
    """
    
    # -------------------------------------------------------------------------
    # PART 1: INITIALIZATION (Ported from sgp4init)
    # -------------------------------------------------------------------------
    
    # Constants
    temp4 = 1.5e-12
    x2o3  = 2.0 / 3.0
    
    # Calculate derived earth constants
    ss = 78.0 / consts.radiusearthkm + 1.0
    qzms2ttemp = (120.0 - 78.0) / consts.radiusearthkm
    qzms2t = qzms2ttemp**4
    
    # Call initl (Assuming functional signature)
    # Note: initl generally converts Kozai mean motion to Brouwer mean motion
    # and handles recover/sanity checks.
    # We pass 'n' for method and 'i' (improved) or 'a' for opsmode as needed.
    # Assuming 'i' (improved) for this implementation.
    (
        no_unkozai, method, ainv, ao, con41, con42, cosio,
        cosio2, eccsq, omeosq, posq,
        rp, rteosq, sinio, gsto
    ) = initl(
        consts.xke, consts.j2, ecco, torch.zeros_like(tsince), # Epoch not needed for init logic usually
        inclo, no_kozai, 'i', 'n'
    )

    # Initial calculations
    a    = torch.pow(no_unkozai * consts.tumin, (-2.0/3.0))
    alta = a * (1.0 + ecco) - 1.0
    altp = a * (1.0 - ecco) - 1.0
    
    # --- Handling 'isimp' (Near Earth vs Deep Space) Logic ---
    # SGP4 usually branches here. To be functional, we calculate values for 
    # the complex path and mask them out if we are in the "simple" case.
    
    # Flag for calculation validity (omeosq >= 0)
    valid_orbit = (omeosq >= 0.0) | (no_unkozai >= 0.0)
    
    # Flag for "Is Simple" (Near Earth simplification)
    # isimp = 1 if rp < (220/R + 1)
    is_simple_mask = rp < (220.0 / consts.radiusearthkm + 1.0)
    
    # Complex perigee calculations
    sfour  = ss
    qzms24 = qzms2t
    perige = (rp - 1.0) * consts.radiusearthkm
    
    # Modification of sfour/qzms24 based on perige
    # Original: if perige < 156: ... if perige < 98: ...
    mask_p156 = perige < 156.0
    mask_p98  = perige < 98.0
    
    sfour_p   = perige - 78.0
    sfour_p   = torch.where(mask_p98, torch.tensor(20.0, device=tsince.device, dtype=tsince.dtype), sfour_p)
    
    qzms24_p_temp = (120.0 - sfour_p) / consts.radiusearthkm
    qzms24_p      = qzms24_p_temp**4
    sfour_p       = sfour_p / consts.radiusearthkm + 1.0
    
    # Apply perigee adjustments
    sfour  = torch.where(mask_p156, sfour_p, sfour)
    qzms24 = torch.where(mask_p156, qzms24_p, qzms24)

    # Calculate variables required for secular/long-period corrections
    pinvsq = 1.0 / posq
    tsi    = 1.0 / (ao - sfour)
    eta    = ao * ecco * tsi
    etasq  = eta * eta
    eeta   = ecco * eta
    psisq  = torch.abs(1.0 - etasq)
    coef   = qzms24 * torch.pow(tsi, 4.0)
    coef1  = coef / torch.pow(psisq, 3.5)
    
    cc2    = coef1 * no_unkozai * (ao * (1.0 + 1.5 * etasq + eeta *
             (4.0 + etasq)) + 0.375 * consts.j2 * tsi / psisq * con41 *
             (8.0 + 3.0 * etasq * (8.0 + etasq)))
             
    cc1    = bstar * cc2
    
    cc3    = torch.where(
        ecco > 1.0e-4, 
        -2.0 * coef * tsi * consts.j3oj2 * no_unkozai * sinio / ecco, 
        torch.tensor(0.0, device=tsince.device, dtype=tsince.dtype)
    )
    
    x1mth2 = 1.0 - cosio2
    
    cc4    = 2.0 * no_unkozai * coef1 * ao * omeosq * \
             (eta * (2.0 + 0.5 * etasq) + ecco *
             (0.5 + 2.0 * etasq) - consts.j2 * tsi / (ao * psisq) *
             (-3.0 * con41 * (1.0 - 2.0 * eeta + etasq *
             (1.5 - 0.5 * eeta)) + 0.75 * x1mth2 *
             (2.0 * etasq - eeta * (1.0 + etasq)) * torch.cos(2.0 * argpo)))
             
    cc5    = 2.0 * coef1 * ao * omeosq * (1.0 + 2.75 * (etasq + eeta) + eeta * etasq)
    
    cosio4 = cosio2 * cosio2
    temp1  = 1.5 * consts.j2 * pinvsq * no_unkozai
    temp2  = 0.5 * temp1 * consts.j2 * pinvsq
    temp3  = -0.46875 * consts.j4 * pinvsq * pinvsq * no_unkozai
    
    mdot     = no_unkozai + 0.5 * temp1 * rteosq * con41 + 0.0625 * \
               temp2 * rteosq * (13.0 - 78.0 * cosio2 + 137.0 * cosio4)
               
    argpdot  = (-0.5 * temp1 * con42 + 0.0625 * temp2 *
                (7.0 - 114.0 * cosio2 + 395.0 * cosio4) +
                temp3 * (3.0 - 36.0 * cosio2 + 49.0 * cosio4))
                
    xhdot1   = -temp1 * cosio
    nodedot  = xhdot1 + (0.5 * temp2 * (4.0 - 19.0 * cosio2) +
                2.0 * temp3 * (3.0 - 7.0 * cosio2)) * cosio
                
    omgcof   = bstar * cc3 * torch.cos(argpo)
    
    xmcof    = torch.where(
        ecco > 1.0e-4,
        -x2o3 * coef * bstar / eeta,
        torch.tensor(0.0, device=tsince.device, dtype=tsince.dtype)
    )
    
    nodecf   = 3.5 * omeosq * xhdot1 * cc1
    t2cof    = 1.5 * cc1
    
    # xlcoff logic
    xlcof_val = -0.25 * consts.j3oj2 * sinio * (3.0 + 5.0 * cosio) / (1.0 + cosio)
    xlcof_alt = -0.25 * consts.j3oj2 * sinio * (3.0 + 5.0 * cosio) / temp4
    xlcof = torch.where(torch.abs(cosio + 1.0) > 1.5e-12, xlcof_val, xlcof_alt)
    
    aycof    = -0.5 * consts.j3oj2 * sinio
    delmo    = (1.0 + eta * torch.cos(mo))**3
    sinmao   = torch.sin(mo)
    x7thm1   = 7.0 * cosio2 - 1.0

    # Drag coefficients (d2, d3, d4)
    # These are only used if NOT simple. We compute them, then mask them.
    cc1sq = cc1 * cc1
    d2    = 4.0 * ao * tsi * cc1sq
    temp  = d2 * tsi * cc1 / 3.0
    d3    = (17.0 * ao + sfour) * temp
    d4    = 0.5 * temp * ao * tsi * (221.0 * ao + 31.0 * sfour) * cc1
    t3cof = d2 + 2.0 * cc1sq
    t4cof = 0.25 * (3.0 * d3 + cc1 * (12.0 * d2 + 10.0 * cc1sq))
    t5cof = 0.2 * (3.0 * d4 + 12.0 * cc1 * d3 + 6.0 * d2 * d2 + 15.0 * cc1sq * (2.0 * d2 + cc1sq))

    # Apply the "Simple" mask. 
    # If is_simple_mask is True, these specific coefficients should be 0 
    # so they don't affect the propagation math below.
    d2    = torch.where(is_simple_mask, torch.tensor(0.0, device=tsince.device), d2)
    d3    = torch.where(is_simple_mask, torch.tensor(0.0, device=tsince.device), d3)
    d4    = torch.where(is_simple_mask, torch.tensor(0.0, device=tsince.device), d4)
    t3cof = torch.where(is_simple_mask, torch.tensor(0.0, device=tsince.device), t3cof)
    t4cof = torch.where(is_simple_mask, torch.tensor(0.0, device=tsince.device), t4cof)
    t5cof = torch.where(is_simple_mask, torch.tensor(0.0, device=tsince.device), t5cof)
    # note: cc1, cc4, cc5 are used regardless of mode in standard SGP4, 
    # but specific terms involving them are gated by d2/d3 checks usually.
    # In the provided sgp4.py, cc1/cc4/cc5 are used unconditionally for linear terms
    # and conditionally for higher order. The conditional logic is handled by d2..t5cof being 0.
    
    # -------------------------------------------------------------------------
    # PART 2: PROPAGATION (Ported from sgp4.py)
    # -------------------------------------------------------------------------
    
    # Secular gravity and atmospheric drag
    xmdf   = mo + mdot * tsince
    argpdf = argpo + argpdot * tsince
    nodedf = nodeo + nodedot * tsince
    argpm  = argpdf
    mm     = xmdf
    t2     = tsince * tsince
    nodem  = nodedf + nodecf * t2
    tempa  = 1.0 - cc1 * tsince
    tempe  = bstar * cc4 * tsince
    templ  = t2cof * t2

    # Higher order drag (If not simple)
    # Since we masked d2...t5cof with zeros if simple, we can run this unconditionally
    delomg = omgcof * tsince
    delmtemp = 1.0 + eta * torch.cos(xmdf)
    delm   = xmcof * (delmtemp**3 - delmo)
    temp   = delomg + delm
    mm     = xmdf + temp
    argpm  = argpdf - temp
    t3     = t2 * tsince
    t4     = t3 * tsince
    tempa  = tempa - d2 * t2 - d3 * t3 - d4 * t4
    tempe  = tempe + bstar * cc5 * (torch.sin(mm) - sinmao)
    templ  = templ + t3cof * t3 + t4 * (t4cof + tsince * t5cof)

    nm    = no_unkozai
    em    = ecco
    inclm = inclo

    # Reconstruct mean elements
    am = torch.pow((consts.xke / nm), x2o3) * tempa * tempa
    nm = consts.xke / torch.pow(am, 1.5)
    em = em - tempe

    # Fix tolerance
    em = torch.clamp(em, min=1.0e-6)
    
    mm  = mm + no_unkozai * templ
    xlm = mm + argpm + nodem
    
    emsq = em * em
    temp = 1.0 - emsq

    # Modulo arithmetic for angles
    nodem = torch.fmod(nodem, 2 * np.pi)
    argpm = torch.fmod(argpm, 2 * np.pi)
    xlm   = torch.fmod(xlm, 2 * np.pi)
    mm    = torch.fmod(xlm - argpm - nodem, 2 * np.pi)

    # Compute extra mean quantities
    sinim = torch.sin(inclm)
    cosim = torch.cos(inclm)
    
    ep    = em
    xincp = inclm
    argpp = argpm
    nodep = nodem
    mp    = mm
    sinip = sinim
    cosip = cosim

    axnl = ep * torch.cos(argpp)
    temp = 1.0 / (am * (1.0 - ep * ep))
    aynl = ep * torch.sin(argpp) + temp * aycof
    xl   = mp + argpp + nodep + temp * xlcof * axnl

    # Solve Kepler's equation
    u    = torch.fmod(xl - nodep, 2 * np.pi)
    eo1  = u
    
    # Fixed iteration loop for JIT/vmap stability
    for _ in range(10):
        coseo1 = torch.cos(eo1)
        sineo1 = torch.sin(eo1)
        tem5   = 1.0 - coseo1 * axnl - sineo1 * aynl
        tem5   = (u - aynl * coseo1 + axnl * sineo1 - eo1) / tem5
        # Clamp adjustment to prevent instability
        tem5   = torch.clamp(tem5, -0.95, 0.95)
        eo1    = eo1 + tem5
    
    # Short period preliminary quantities
    coseo1 = torch.cos(eo1)
    sineo1 = torch.sin(eo1)
    
    ecose = axnl * coseo1 + aynl * sineo1
    esine = axnl * sineo1 - aynl * coseo1
    el2   = axnl * axnl + aynl * aynl
    pl    = am * (1.0 - el2)
    
    # Warning: Standard SGP4 checks if pl < 0 here and sets error. 
    # We proceed blindly for differentiability, but results may be garbage if pl < 0.

    rl     = am * (1.0 - ecose)
    rdotl  = torch.sqrt(am) * esine / rl
    rvdotl = torch.sqrt(pl) / rl
    betal  = torch.sqrt(1.0 - el2)
    temp   = esine / (1.0 + betal)
    sinu   = am / rl * (sineo1 - aynl - axnl * temp)
    cosu   = am / rl * (coseo1 - axnl + aynl * temp)
    su     = torch.atan2(sinu, cosu)
    sin2u  = (cosu + cosu) * sinu
    cos2u  = 1.0 - 2.0 * sinu * sinu
    temp   = 1.0 / pl
    temp1  = 0.5 * consts.j2 * temp
    temp2  = temp1 * temp

    mrt   = rl * (1.0 - 1.5 * temp2 * betal * con41) + \
            0.5 * temp1 * x1mth2 * cos2u
            
    su    = su - 0.25 * temp2 * x7thm1 * sin2u
    xnode = nodep + 1.5 * temp2 * cosip * sin2u
    xinc  = xincp + 1.5 * temp2 * cosip * sinip * cos2u
    mvt   = rdotl - nm * temp1 * x1mth2 * sin2u / consts.xke
    rvdot = rvdotl + nm * temp1 * (x1mth2 * cos2u + 1.5 * con41) / consts.xke

    # Orientation vectors
    sinsu = torch.sin(su)
    cossu = torch.cos(su)
    snod  = torch.sin(xnode)
    cnod  = torch.cos(xnode)
    sini  = torch.sin(xinc)
    cosi  = torch.cos(xinc)
    
    xmx   = -snod * cosi
    xmy   = cnod * cosi
    ux    = xmx * sinsu + cnod * cossu
    uy    = xmy * sinsu + snod * cossu
    uz    = sini * sinsu
    vx    = xmx * cossu - cnod * sinsu
    vy    = xmy * cossu - snod * sinsu
    vz    = sini * cossu

    # Position and velocity (in km and km/sec)
    vkmpersec = consts.radiusearthkm * consts.xke / 60.0
    mr        = mrt * consts.radiusearthkm

    # Pack results: [Batch, 3]
    rx = mr * ux
    ry = mr * uy
    rz = mr * uz
    
    vx_val = (mvt * ux + rvdot * vx) * vkmpersec
    vy_val = (mvt * uy + rvdot * vy) * vkmpersec
    vz_val = (mvt * uz + rvdot * vz) * vkmpersec
    
    r = torch.stack([rx, ry, rz], dim=-1)
    v = torch.stack([vx_val, vy_val, vz_val], dim=-1)
    
    return r, v