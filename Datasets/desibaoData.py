import numpy as np
from pathlib import Path
from scipy.linalg import block_diag

# ---- Physical Constants ---- #

# Speed of light in km/s
c = 299792.458 # km/s

# Fiducial Sound Horizon at drag epoch (Mpc) -- estimated, not observed
# Based on Planck 2018 TT,TE,EE+lowE+lensing
r_d_fid = 147.09 # Mpc

# ---- Cosmological Functions ---- #
def H_fromDH_over_rs(DH_over_rs : float, speed_of_light : float = c) -> float:
    """
    Compute the parameter from the ratio of the radial distance to the bubble radius at
    the drag epoch.
    
    This comes from the relationship
    D_H = (c * rs) / H
    """
    return speed_of_light / DH_over_rs

# ---- File Names ---- #
tracer_names : list = list([
    "BGS", "LRG1", "LRG2", "LRG+ELG", "ELG", "Lya", "QSO"
])

data_paths : dict = dict({
    "BGS":      Path("Datasets/desibaoData/desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_mean.txt"),
    "LRG1":     Path("Datasets/desibaoData/desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_mean.txt"),
    "LRG2":     Path("Datasets/desibaoData/desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_mean.txt"),
    "LRG+ELG":  Path("Datasets/desibaoData/desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_mean.txt"),
    "ELG":      Path("Datasets/desibaoData/desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_mean.txt"),
    "Lya":      Path("Datasets/desibaoData/desi_2024_gaussian_bao_Lya_GCcomb_mean.txt"),
    "QSO":      Path("Datasets/desibaoData/desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_mean.txt"),
})

covmat_paths : dict = dict({
    "BGS":      Path("Datasets/desibaoData/desi_2024_gaussian_bao_BGS_BRIGHT-21.5_GCcomb_z0.1-0.4_cov.txt"),
    "LRG1":     Path("Datasets/desibaoData/desi_2024_gaussian_bao_LRG_GCcomb_z0.4-0.6_cov.txt"),
    "LRG2":     Path("Datasets/desibaoData/desi_2024_gaussian_bao_LRG_GCcomb_z0.6-0.8_cov.txt"),
    "LRG+ELG":  Path("Datasets/desibaoData/desi_2024_gaussian_bao_LRG+ELG_LOPnotqso_GCcomb_z0.8-1.1_cov.txt"),
    "ELG":      Path("Datasets/desibaoData/desi_2024_gaussian_bao_ELG_LOPnotqso_GCcomb_z1.1-1.6_cov.txt"),
    "Lya":      Path("Datasets/desibaoData/desi_2024_gaussian_bao_Lya_GCcomb_cov.txt"),
    "QSO":      Path("Datasets/desibaoData/desi_2024_gaussian_bao_QSO_GCcomb_z0.8-2.1_cov.txt"),
})

# ---- Loading Functions ---- #
def load_data(path: Path):
    try:
        # np.atleast_1d ensures even single-row files act like arrays
        return np.atleast_1d(np.loadtxt(path, dtype={'names': ('z', 'value', 'quantity'), 'formats': ('f8', 'f8', 'U20')}))
    except FileNotFoundError:
        print(f"[Warning]: Data file {path} not found.")
        return None

def load_covmat(path: Path):
    try:
        # np.atleast_2d ensures single-value covmats act like matrices
        return np.atleast_2d(np.loadtxt(path, dtype='f8'))
    except FileNotFoundError:
        print(f"[Warning]: Covmat file {path} not found.")
        return None
    
# ---- Data Class ---- #

class DESI_BAO:
    def __init__(self):
        self.z = None       # Redshifts
        self.H = None       # H(z) values extracted from DH/rd
        self.covmat = None  # Covariance matrix for H(z)
        self.std_err = None # Diagonal standard errors for H(z)
        self.inv_cov = None # Inverse covariance
        
        self.process_data()

    def process_data(self):
        # Temp lists
        z_list = []
        dh_rd_list = []  # Stores the raw DH/rd values
        cov_list = []    # Stores the raw covariances

        for name in data_paths.keys():
            raw_tracer = load_data(data_paths[name])
            raw_covmat = load_covmat(covmat_paths[name])

            if raw_tracer is None or raw_covmat is None:
                continue

            mask = raw_tracer['quantity'] == 'DH_over_rs' # Pull radial component only

            if np.any(mask):
                z_chunk = raw_tracer['z'][mask]
                val_chunk = raw_tracer['value'][mask]
                
                # Extract sub-covariance matrix
                cov_chunk = raw_covmat[np.ix_(mask, mask)]
                
                z_list.append(z_chunk)
                dh_rd_list.append(val_chunk)
                cov_list.append(cov_chunk)
                
                print(f"[DESI]: Loaded {len(z_chunk)} points from {name}")
        
        # -- Concat data -- #

        if not z_list:
            raise ValueError("[DESI]: No data loaded!")
        
        self.z = np.concatenate(z_list)
        raw_values = np.concatenate(dh_rd_list)
        raw_covmat = block_diag(*cov_list)

        # -- Physics -- #
        # Relation: (DH / rd) = c / (H * rd)

        self.H = c / (raw_values * r_d_fid)

        # -- Propagate Covariance - Jacobian Transformation -- #
        """
        # y = DH/rd
        # H = A / y   where A = c / rd, a constant
        # dH/dy = - A / y^2 = - H / y
        """

        jacobian_diag = - self.H / raw_values
        J = np.diag(jacobian_diag)
        
        # C_H = J * C_y * J.T
        self.covmat = J @ raw_covmat @ J.T

        self.std_err = np.sqrt(np.diag(self.covmat))

        try:
            self.inv_cov = np.linalg.inv(self.covmat)
        except np.linalg.LinAlgError:
            print("[DESI]: Inversion error, adding jitter...")
            jitter = 1e-12 * np.mean(np.diag(self.covmat))
            self.inv_cov = np.linalg.inv(self.covmat + jitter * np.eye(len(self.H)))

def getDESIData() -> DESI_BAO:
    return DESI_BAO()

"""
# ---- Data Extraction ---- #
DH_filter : dict = dict({})

for name in data_paths.keys():
    raw_tracer : np.ndarray = load_data(data_paths[name])
    raw_covmat : np.ndarray = load_covmat(covmat_paths[name])
    
    # Skip if None
    if raw_tracer is None or raw_covmat is None:
        continue
    
    mask = raw_tracer['quantity'] == 'DH_over_rs'
    
    if np.any(mask):
        DH_filter[name] = {
            'data': raw_tracer[mask],
            'cov': raw_covmat[np.ix_(mask, mask)]
        }

# ---- Verification ---- #
print(f"Filtered {len(DH_filter)} tracers containing 'DH_over_rs':")
for name, content in DH_filter.items():
    z_values = content['data']['z']
    print(f"- {name:8} | Points: {len(z_values)} | Redshifts: {z_values}")

# ---- Concat Covmats ---- #
cum_redshifts : list = list([])
cum_values_DH : list = list([])
cum_covmats : list = list([])

for name in DH_filter:
    cum_redshifts.append(DH_filter[name]['data']['z'])
    cum_values_DH.append(DH_filter[name]['data']['value'])
    cum_covmats.append(DH_filter[name]['data']['cov'])

# Super Vectors
concat_values = np.concatenate(cum_values_DH)
concat_redshifts = np.concatenate(cum_redshifts)

# Super covariance matrix
Ccovmat = block_diag(*cum_covmats)
print(DH_filter['LRG1']['data']['z'])

# ---- Compute Parameters ---- #
"""
#We do a little cosmology here :D
"""
H_list : list = list([])
for ratio in cum_values_DH:
    H_list.append(H_fromDH_over_rs(ratio))
    
cum_H : np.ndarray = np.array(H_list)


"""


# Example of how to access a specific one:
# lrg_data = filtered_results['LRG1']['data']
# lrg_cov  = filtered_results['LRG1']['cov']