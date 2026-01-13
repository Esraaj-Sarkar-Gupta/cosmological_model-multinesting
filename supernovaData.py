import numpy as np
from pathlib import Path

# ---- Import Data ---- #
"""
Import binned Pantheon+ data

Reference: 
"""

DATA_PATH   : Path = Path("binnedpantheonplusData/lcparam_DS17f.txt")
COVMAT_PATH : Path = Path("binnedpantheonplusData/sys_DS17f.txt")

# ---- Physics ---- #

# ---- Load Data ---- #
def load_data(data_path : Path, covmat_path : Path):
    print("[Pantheon]: Loading data...")

    # Load Binned Observed Data
    raw_data = np.loadtxt(data_path, comments="#")
    z = raw_data[:, 1]
    mb = raw_data[:, 4]
    dmb = raw_data[:, 5]

    # Load Systemic Covariance Matrix
    with open(covmat_path, 'r') as f:
                lines = f.readlines()
                n_bins = int(lines[0])
                cov_values = []
                for line in lines[1:]:
                    cov_values.extend([float(x) for x in line.split()])
            
    # Reshape
    C_sys = np.array(cov_values).reshape((n_bins, n_bins))
    
    # Total Covariance = Systematic + Statistical
    C_stat = np.diag(dmb**2) # Diagonal covmat
    covmat = C_sys + C_stat
    
    # 4. Invert
    inv_covmat = np.linalg.inv(covmat)
    sign, logdet = np.linalg.slogdet(covmat)
    
    print(f"[Pantheon]: Loaded {len(z)} binned supernovae.")

    return z, mb, dmb, covmat, inv_covmat, logdet



# ---- Data Class ---- #

class PantheonPlusData:
    def __init__(self):
        (self.z, self.mb, self.dmb,
        self.covmat, self.inv_covmat, self.logdet) = load_data(
             DATA_PATH, COVMAT_PATH   
        )

# ---- Export Function ---- #

def getPantheonPlusData() -> PantheonPlusData:
      return PantheonPlusData()


# Reference: https://github.com/dscolnic/Pantheon/blob/master/Binned_data/sys_DS17f.txt