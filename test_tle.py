import torch
import datetime
from diffod.tle import tle_encode, tle_decode, batch_decode, batch_encode

def _parse_tle_epoch(year_str: str, days_str: str) -> datetime.datetime:
    """Helper to extract datetime from TLE for the encoder input."""
    two_digit_yr = int(year_str)
    year = 2000 + two_digit_yr if two_digit_yr < 57 else 1900 + two_digit_yr
    days = float(days_str)
    start_of_year = datetime.datetime(year - 1, 12, 31)
    return start_of_year + datetime.timedelta(days=days)

def run_tests():
    # ---------------------------------------------------------
    # 1. Define Test Data
    # ---------------------------------------------------------
    # TLE 1: ISS (Zarya)
    tle1 = [
        "1 25544U 98067A   20316.40015046  .00001878  00000-0  44436-4 0  9997",
        "2 25544  51.6465 289.4354 0001961 270.2184  89.8601 15.49504104255152"
    ]
    tle1_epoch = _parse_tle_epoch("20", "316.40015046")
    tle1_satnum = 25544

    # TLE 2: Random debris/satellite
    tle2 = [
        "1 43013U 17071A   21303.41500000  .00000112  00000-0  11234-4 0  9993",
        "2 43013  97.1234 100.5555 0011234 150.1234 210.9876 15.12345678123456"
    ]
    tle2_epoch = _parse_tle_epoch("21", "303.41500000")
    tle2_satnum = 43013

    # ---------------------------------------------------------
    # 2. Test Single TLE
    # ---------------------------------------------------------
    print("=== TESTING SINGLE TLE ===")
    
    # Decode: Pass the flat list of strings directly
    tensor_single = tle_decode(tle1)
    print(f"Decoded Tensor Shape (Single): {tensor_single.shape}") # Should be [9]
    
    # Encode: Extract the individual float values using .item()
    single_output = tle_encode(
        bstar=tensor_single[0].item(),
        ndot=tensor_single[1].item(),
        nddot=tensor_single[2].item(),
        ecco=tensor_single[3].item(),
        argpo=tensor_single[4].item(),
        inclo=tensor_single[5].item(),
        mo=tensor_single[6].item(),
        no_kozai=tensor_single[7].item(),
        nodeo=tensor_single[8].item(),
        sat_num=tle1_satnum,
        epoch=tle1_epoch
    )
    
    # Compare
    print("\nOriginal vs Reconstructed (Single):")
    print(f"Orig L1: {tle1[0]}")
    print(f"Recon L1:{single_output[0]}")
    print(f"Orig L2: {tle1[1]}")
    print(f"Recon L2:{single_output[1]}")


    # ---------------------------------------------------------
    # 3. Test Batch of TLEs
    # ---------------------------------------------------------
    print("\n=== TESTING BATCH OF TLEs ===")
    batch_input = [tle1, tle2]
    batch_epochs = [tle1_epoch, tle2_epoch]
    batch_satnums = [tle1_satnum, tle2_satnum]
    
    # Decode
    tensor_batch = batch_decode(batch_input)
    print(f"Decoded Tensor Shape (Batch): {tensor_batch.shape}") # Should be [2, 9]
    
    # Encode: Pass the 1D tensor slices for each parameter
    batch_output = batch_encode(
        bstar=tensor_batch[:, 0],
        ndot=tensor_batch[:, 1],
        nddot=tensor_batch[:, 2],
        ecco=tensor_batch[:, 3],
        argpo=tensor_batch[:, 4],
        inclo=tensor_batch[:, 5],
        mo=tensor_batch[:, 6],
        no_kozai=tensor_batch[:, 7],
        nodeo=tensor_batch[:, 8],
        sat_nums=batch_satnums,
        epochs=batch_epochs
    )

    # Compare
    print("\nBatch Reconstructions:")
    for i in range(len(batch_input)):
        print(f"\n--- Item {i} ---")
        print(f"Orig L1: {batch_input[i][0]}")
        print(f"Recon L1:{batch_output[i][0]}")
        print(f"Orig L2: {batch_input[i][1]}")
        print(f"Recon L2:{batch_output[i][1]}")
        
        # Simple structural assertion 
        assert len(batch_output[i][0]) == 69, f"Line 1 length is {len(batch_output[i][0])}, expected 69"
        assert len(batch_output[i][1]) == 69, f"Line 2 length is {len(batch_output[i][1])}, expected 69"

if __name__ == "__main__":
    run_tests()