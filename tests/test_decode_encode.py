import pytest
import torch
import datetime
from diffod.tle import tle_decode, tle_encode, batch_decode, batch_encode

# --- Helper Functions ---
def compute_checksum(line: str) -> str:
    """Computes the expected checksum digit for a TLE line."""
    return str(sum((int(c) if c.isdigit() else c == '-') for c in line[0:68]) % 10)

def parse_tle_epoch(year_str: str, days_str: str) -> datetime.datetime:
    """Extracts a datetime object for the encoder input."""
    two_digit_yr = int(year_str)
    year = 2000 + two_digit_yr if two_digit_yr < 57 else 1900 + two_digit_yr
    days = float(days_str)
    start_of_year = datetime.datetime(year - 1, 12, 31)
    return start_of_year + datetime.timedelta(days=days)

# --- Test Data ---
TLE_ISS = [
    "1 25544U 98067A   20316.40015046  .00001878  00000-0  44436-4 0  9997",
    "2 25544  51.6465 289.4354 0001961 270.2184  89.8601 15.49504104255152"
]
TLE_ISS_SATNUM = 25544
TLE_ISS_EPOCH = parse_tle_epoch("20", "316.40015046")

TLE_DEBRIS = [
    "1 43013U 17071A   21303.41500000  .00000112  00000-0  11234-4 0  9993",
    "2 43013  97.1234 100.5555 0011234 150.1234 210.9876 15.12345678123456"
]
TLE_DEBRIS_SATNUM = 43013
TLE_DEBRIS_EPOCH = parse_tle_epoch("21", "303.41500000")

@pytest.fixture(autouse=True)
def set_default_dtype():
    torch.set_default_dtype(torch.float64)

# --- Tests ---

@pytest.mark.parametrize("tle_str, sat_num, epoch", [
    (TLE_ISS, TLE_ISS_SATNUM, TLE_ISS_EPOCH),
    (TLE_DEBRIS, TLE_DEBRIS_SATNUM, TLE_DEBRIS_EPOCH)
], ids=["ISS", "Debris"])
def test_single_tle_io(tle_str, sat_num, epoch):
    """Verifies decoding and encoding for a single TLE."""
    # 1. Decode original string
    tensor_orig = tle_decode(tle_str)
    assert tensor_orig.shape == (9,), "Decoded single tensor should be 1D with 9 elements"

    # 2. Encode back to string
    encoded_lines = tle_encode(
        bstar=tensor_orig[0].item(), ndot=tensor_orig[1].item(), nddot=tensor_orig[2].item(),
        ecco=tensor_orig[3].item(), argpo=tensor_orig[4].item(), inclo=tensor_orig[5].item(),
        mo=tensor_orig[6].item(), no_kozai=tensor_orig[7].item(), nodeo=tensor_orig[8].item(),
        sat_num=sat_num, epoch=epoch
    )

    # 3. Structural validation
    l1, l2 = encoded_lines[0], encoded_lines[1]
    assert len(l1) == 69, f"Line 1 length is {len(l1)}, expected 69"
    assert len(l2) == 69, f"Line 2 length is {len(l2)}, expected 69"
    assert l1[-1] == compute_checksum(l1), "Line 1 checksum is invalid"
    assert l2[-1] == compute_checksum(l2), "Line 2 checksum is invalid"

    # 4. Data preservation validation (Round-trip)
    tensor_recon = tle_decode(encoded_lines)
    assert torch.allclose(tensor_orig, tensor_recon, atol=1e-8), "Data lost during encode/decode round-trip"


def test_batch_tle_io():
    """Verifies decoding and encoding for a batch of TLEs."""
    tles = [TLE_ISS, TLE_DEBRIS]
    sat_nums = [TLE_ISS_SATNUM, TLE_DEBRIS_SATNUM]
    epochs = [TLE_ISS_EPOCH, TLE_DEBRIS_EPOCH]
    
    # 1. Decode batch
    tensor_batch = batch_decode(tles)
    assert tensor_batch.shape == (2, 9), "Decoded batch tensor should be 2D (N, 9)"

    # 2. Encode batch
    encoded_batch = batch_encode(
        bstar=tensor_batch[:, 0], ndot=tensor_batch[:, 1], nddot=tensor_batch[:, 2],
        ecco=tensor_batch[:, 3], argpo=tensor_batch[:, 4], inclo=tensor_batch[:, 5],
        mo=tensor_batch[:, 6], no_kozai=tensor_batch[:, 7], nodeo=tensor_batch[:, 8],
        sat_nums=sat_nums, epochs=epochs
    )

    assert len(encoded_batch) == 2, "Encoder should return a list of 2 TLE string pairs"

    # 3. Structural & Data Validation per item
    for i in range(2):
        l1, l2 = encoded_batch[i][0], encoded_batch[i][1]
        
        # Structural
        assert len(l1) == 69, f"Batch item {i} Line 1 length is {len(l1)}, expected 69"
        assert len(l2) == 69, f"Batch item {i} Line 2 length is {len(l2)}, expected 69"
        assert l1[-1] == compute_checksum(l1), f"Batch item {i} Line 1 checksum invalid"
        assert l2[-1] == compute_checksum(l2), f"Batch item {i} Line 2 checksum invalid"

    # 4. Data preservation validation
    tensor_batch_recon = batch_decode(encoded_batch)
    assert torch.allclose(tensor_batch, tensor_batch_recon, atol=1e-8), "Data lost in batch round-trip"