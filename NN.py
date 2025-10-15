# NN.py File

import pathlib, pickle
from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn as nn
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import warnings
import copy 
from scipy.interpolate import RBFInterpolator
from matplotlib.gridspec import GridSpec
import matplotlib.tri as tri
from matplotlib.patches import Polygon
from matplotlib.cm import ScalarMappable
import math
import matplotlib.path as mpath
import stat
import subprocess
import pathlib
from pathlib import Path
import shutil
import re
import warnings
from sklearn.exceptions import InconsistentVersionWarning

device = "cpu"
plt.rcParams.update({
    "text.usetex": False,
    "font.family": "serif",
    "font.size": 14,
})

def replace_in_file(file_path, replacements_dict, regex_flags=0):
    """Same helper you used before: regex replace over entire file."""
    with open(file_path, 'r') as f:
        content = f.read()

    for pattern, replacement in replacements_dict.items():
        content = re.sub(pattern, str(replacement), content, flags=regex_flags)

    with open(file_path, 'w') as f:
        f.write(content)

def update_blockMesh_from_source(H1, H2, H3, H4,
                                 Y2, Y3, Y4,
                                 theta1):
    """
    Copy blockMeshDict_source into system/blockMeshDict for the given case,
    then patch only H1–H4 and Y2–Y4 in the new file, using the same anchored
    regex approach as your working code.

    Parameters
    ----------
    case_dir : str or Path
        Path to the case directory (e.g., 'NN_files/dummy_case_mesh_coordinates').
    H1, H2, H3, H4 : float
    Y2, Y3, Y4     : float
    """
    case_dir = Path("NN_files/dummy_case_mesh_coordinates")

    src = case_dir / "blockMeshDict_source"
    dst_dir = case_dir / "system"
    dst = dst_dir / "blockMeshDict"

    if not src.exists():
        raise FileNotFoundError(f"Could not find source file: {src}")

    # Ensure system/ exists; then copy (overwrite) the target
    dst_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    # Patch only the requested parameters, using your exact pattern style.
    replacements = {
        r'^H_1 H1;': f'H_1 {H1};',
        r'^H_2 H2;': f'H_2 {H2};',
        r'^H_3 H3;': f'H_3 {H3};',
        r'^H_4 H4;': f'H_4 {H4};',
        r'^Y_2 Y2;': f'Y_2 {Y2};',
        r'^Y_3 Y3;': f'Y_3 {Y3};',
        r'^Y_4 Y4;': f'Y_4 {Y4};',
        r'^theta_1 theta1;': f'theta_1 {theta1};',
        # NOTE: intentionally not touching theta_1 here.
    }
    replace_in_file(dst, replacements, regex_flags=re.MULTILINE)

    print(f"✓ Copied '{src}' → '{dst}' and patched H1–H4, Y2–Y4.")

def run_blockmesh_output():
    # Hardcoded path (must exist)
    p = pathlib.Path("NN_files/dummy_case_mesh_coordinates/run_blockmesh_outputC")
    st = os.stat(p)
    if not (st.st_mode & stat.S_IXUSR):
        os.chmod(p, st.st_mode | stat.S_IXUSR)
    subprocess.run(["./" + p.name], cwd=str(p.parent), check=True)

def extract_trunk_spatial_fixed(num_nodes=3780):
    """Return (1, num_nodes, 2) array with [x, y] coordinates from constant/C."""
    C_path = os.path.join("NN_files", "dummy_case_mesh_coordinates", "constant", "C")

    with open(C_path, "r") as f:
        lines = f.readlines()
    if int(lines[21].strip()) != num_nodes:
        raise ValueError("num_nodes mismatch between C and solver files.")

    coord_lines = lines[23 : 23 + num_nodes]
    coords = np.array(
        [line.strip()[1:-1].split()[:2] for line in coord_lines],
        dtype=np.float32
    )
    return coords.reshape(1, num_nodes, 2)

class CPNN(nn.Module):
    def __init__(self, coord_cfg, branch_load_cfg, fcnn_cfg):
        super().__init__()
        self.layers_trunk_coordinates = self._stack(coord_cfg)
        self.layers_branch_loading = self._stack(branch_load_cfg)
        self.layers_fcnn = self._stack(fcnn_cfg, last=False)

    def forward(self, u, y):
        z = (self.layers_trunk_coordinates(y) * self.layers_branch_loading(u))
        return self.layers_fcnn(z)

    @staticmethod
    def _stack(cfg, last=True):
        layers = OrderedDict()
        for i in range(len(cfg) - 1):
            layers[f'layer_{i}'] = nn.Linear(cfg[i], cfg[i + 1])
            if i < len(cfg) - 2 or last:
                layers[f'act_{i}'] = nn.Tanh()
        return nn.Sequential(layers)


def load_model():
    """Load the CPNN model with pre-trained weights."""
    best_layer = 92
    enc_layers = 2
    fcnn_layers = 6

    # branch_load_cfg = [13] + [best_layer] * enc_layers
    branch_load_cfg = [14] + [best_layer] * enc_layers
    coord_cfg = [2] + [best_layer] * enc_layers
    fcnn_cfg = [best_layer] * fcnn_layers + [1]

    model = CPNN(coord_cfg, branch_load_cfg, fcnn_cfg).to(device)
    
    # Fixed: map_location and weights_only go INSIDE torch.load()
    # state_dict = torch.load("NN_files/best_val_opt_param_561.pt",
    #                        map_location='cpu',
    #                        weights_only=True)
    state_dict = torch.load("NN_files/best_val_opt_param_800.pt",
                        map_location='cpu',
                        weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_scalers():
    """Load the pre-trained scalers."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", InconsistentVersionWarning)
        with open("NN_files/scalers/u_scaler.pkl", "rb") as f: 
            u_scaler = pickle.load(f)
        with open("NN_files/scalers/y_scaler.pkl", "rb") as f: 
            y_scaler = pickle.load(f)
        with open("NN_files/scalers/s_scaler.pkl", "rb") as f: 
            s_scaler = pickle.load(f)
    return u_scaler, y_scaler, s_scaler


def mask_irregular_polygon(coordinates, H2, H3, H4, H5, Y2, Y4, Theta1, ymax):
    """
    coordinates : (N, 3) array – you generate this just before calling the NN
    Returns     : Boolean mask where True  -> point is **kept**
                                      False -> point is inside a dead-zone
    """
    # dead-zone #1 – irregular quadrilateral
    poly_vertices_quad = np.array([
        [H5, 0.0],
        [H4, 0.0],
        [H4, Y4 + (H4 - H2) * math.tan(math.radians(Theta1))],
        [H2, Y4]
    ])
    deadzone_path_quad = mpath.Path(poly_vertices_quad)

    # dead-zone #2 – small rectangle
    poly_vertices_rect = np.array([
        [H3, ymax - Y2],
        [H3 + 0.04, ymax - Y2],
        [H3 + 0.04, ymax],
        [H3, ymax]
    ])
    deadzone_path_rect = mpath.Path(poly_vertices_rect)

    # point-in-polygon tests (vectorised)
    xy_points = coordinates[:, :2]
    inside_quad = deadzone_path_quad.contains_points(xy_points)
    inside_rect = deadzone_path_rect.contains_points(xy_points)

    mask = ~(inside_quad | inside_rect)
    return mask


def calculate_concentrations(u_data, nn_model, nn_u_scaler, nn_y_scaler, nn_s_scaler, nn_y_indices):
    """
    Main function to calculate ESS and RAS average concentrations.
    
    Parameters:
    -----------
    trunk_indices_file : str
        Path to trunk_indices_{num_points}_points.npz
    branch_loading_file : str
        Path to branch_loading_{num_cases}_cases.npz
    trunk_coordinates_file : str
        Path to trunk_coordinates_{num_cases}_cases_{num_points}_points.npz
    
    Returns:
    --------
    ess_avg : float
        ESS Average concentration
    ras_avg : float
        RAS Average concentration
    prediction_data : dict
        Dictionary containing all intermediate data for plotting
    """
    
    y_data_indices = nn_y_indices

    # Load model and scalers
    model = nn_model
    u_scaler, y_scaler, s_scaler = nn_u_scaler, nn_y_scaler, nn_s_scaler
    
    # Prepare tensors
    y_indices_tensor = torch.tensor(
        y_scaler.transform(y_data_indices.reshape(-1, y_data_indices.shape[-1])).reshape(1, -1, 2),
        dtype=torch.float32, device=device
    )

    u_tensor = torch.tensor(
        u_scaler.transform(u_data.reshape(-1, u_data.shape[-1])).reshape(1, 1, u_data.shape[-1]),
        dtype=torch.float32, device=device
    )
    
    # Model prediction
    model.eval()
    with torch.no_grad():
        prediction = model(u_tensor, y_indices_tensor)
    
    prediction_unnorm = s_scaler.inverse_transform(
        prediction.cpu().detach().numpy().reshape(-1, 1)
    )
    
    # Calculate ESS and RAS concentrations
    indices = np.asarray(y_data_indices, dtype=int).reshape(-1, 2)
    vals = np.asarray(prediction_unnorm, dtype=float).reshape(-1)
    idx_to_val = {(i, j): v for (i, j), v in zip(indices, vals)}
    
    # ESS cells: (81,52) and (81,53)
    ess_cells = [(81, 52), (81, 53)]
    ess_values = np.array([idx_to_val[c] for c in ess_cells])
    ess_values_positive = np.maximum(0.0, ess_values)
    ess_avg = np.mean(ess_values_positive)
    
    # RAS cells: (1,1) to (40,1)
    ras_cells = [(i, 1) for i in range(1, 41)]
    ras_values = np.array([idx_to_val[c] for c in ras_cells])
    ras_values_positive = np.maximum(0.0, ras_values)  
    ras_avg = np.mean(ras_values_positive)
    
    # print("ess_positive min/max:", ess_values_positive.min(), ess_values_positive.max())
    # print("ras_positive min/max:", ras_values_positive.min(), ras_values_positive.max())

    # Store data for plotting
    prediction_data = {
        'u_data': u_data,
        'prediction_unnorm': prediction_unnorm,
        'y_data_indices': y_data_indices,
        'ess_values': ess_values,
        'ras_values': ras_values,
        'ess_avg': ess_avg,
        'ras_avg': ras_avg
    }
    
    return ess_avg, ras_avg, prediction_data


def plot_contour(prediction_data, output_dir='contours'):
    """
    Plot contour visualization.
    
    Parameters:
    -----------
    prediction_data : dict
        Dictionary containing prediction data from calculate_concentrations
    output_dir : str
        Directory to save the plot
    """
    u_data = prediction_data['u_data']
    y_data_coordinates = prediction_data['y_data_coordinates']
    prediction_unnorm = prediction_data['prediction_unnorm']
    
    H1, H2, H3, H4, Y2, Y3, Y4, Theta1, Q, Q2, MLSS, V0, r_h_a, r_p_a1 = u_data[0, 0, :]
    
    # Build X-Y grid & live-zone mask
    xmin, xmax = H1, H4
    outlet_depth = 0.1
    H5 = H1 + (H2 - H1) / 2
    ymin, ymax = 0, (Y3 + Y4 + outlet_depth +
                     (H4 - H2) * math.tan(math.radians(Theta1)))
    
    x_range = np.linspace(xmin, xmax, 500)
    y_range = np.linspace(ymin, ymax, 500)
    X, Y = np.meshgrid(x_range, y_range)
    grid_xy = np.vstack([X.ravel(), Y.ravel()]).T
    
    mask = mask_irregular_polygon(grid_xy, H2, H3, H4, H5,
                                  Y2, Y4, Theta1, ymax)
    grid_xy_live = grid_xy[mask]
    
    # Interpolate prediction to the grid
    rbf_pred = RBFInterpolator(
        y_data_coordinates.reshape(-1, 2), prediction_unnorm,
        kernel='thin_plate_spline', neighbors=25)
    pred_grid = rbf_pred(grid_xy_live)
    
    # Plot
    masked_pred = np.zeros(X.size)
    deadzone_colour = '#A0A0A0'
    
    quad_xy = [(H5, 0.0),
               (H4, 0.0),
               (H4, Y4 + (H4 - H2)*math.tan(math.radians(Theta1))),
               (H2, Y4)]
    
    rect_xy = [(H3, ymax - Y2),
               (H3 + 0.04, ymax - Y2),
               (H3 + 0.04, ymax),
               (H3, ymax)]
    
    quad_patch = Polygon(quad_xy, closed=True,
                        facecolor=deadzone_colour, edgecolor=deadzone_colour,
                        zorder=10)
    rect_patch = Polygon(rect_xy, closed=True,
                        facecolor=deadzone_colour, edgecolor=deadzone_colour,
                        zorder=10)
    
    triang = tri.Triangulation(X.ravel(), Y.ravel())
    fig, ax_pred = plt.subplots(1, figsize=(6.5, 3), dpi=300)
    
    masked_pred[mask] = pred_grid[:, 0]
    
    vmin = 0
    vmax = float(np.nanmax(masked_pred))
    if vmin == vmax:
        vmax = vmin
    
    cs = ax_pred.tricontourf(triang, masked_pred, 100, cmap="turbo",
                             vmin=vmin, vmax=vmax)
    fig.colorbar(cs, ax=ax_pred, label='c', location='left')
    
    ax_pred.add_patch(copy.copy(quad_patch))
    ax_pred.add_patch(copy.copy(rect_patch))
    
    plt.tight_layout()
    
    out_dir = os.path.join(output_dir, f'case')
    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(os.path.join(out_dir, 'contour_300levels.png'),
                dpi=300, bbox_inches='tight')
    plt.show()


def print_results(ess_avg, ras_avg, prediction_data):
    """Print ESS and RAS analysis results."""
    ess_values = prediction_data['ess_values']
    ras_values = prediction_data['ras_values']
    
    ess_std = np.std(ess_values, ddof=1)
    ras_std = np.std(ras_values, ddof=1)
    
    print(f"\n{'='*60}")
    print(f"ESS (Effluent Suspended Solids) Analysis:")
    print(f"{'='*60}")
    print(f"  Average concentration: {ess_avg:.6g}")
    print(f"  Standard deviation:    {ess_std:.6g}")
    print(f"  Min concentration:     {np.min(ess_values):.6g}")
    print(f"  Max concentration:     {np.max(ess_values):.6g}")
    print(f"\n{'='*60}")
    print(f"RAS (Return Activated Sludge) Analysis:")
    print(f"{'='*60}")
    print(f"  Average concentration: {ras_avg:.6g}")
    print(f"  Standard deviation:    {ras_std:.6g}")
    print(f"  Min concentration:     {np.min(ras_values):.6g}")
    print(f"  Max concentration:     {np.max(ras_values):.6g}")
    print(f"  Range:                 {np.max(ras_values) - np.min(ras_values):.6g}")
    print(f"{'='*60}\n")


def run_prediction_pipeline(Q_scaled, Q2_scaled, MLSS_scaled, settling_params, nn_model, nn_u_scaler, nn_y_scaler, nn_s_scaler, nn_y_indices, plot=False):
    """
    Runs the full pipeline:
    - (Optionally) load branch loading and coordinates (kept as comments per original)
    - Build u_data2 from given scalars
    - Update and run blockMesh
    - Extract trunk spatial coordinates
    - Calculate concentrations
    - Print results
    - (Optionally) plot contour

    Returns:
        ess_avg, ras_avg
    """

    # base_dir = '/lustre/isaac24/scratch/mshatara/openfoam/mshatarah-v2306/run/Secondary_Clarifier/1000_cases/Data_extracted/concatenated_cases'
    # num_cases = 703
    # num_points = 3780
    # case_number = 1
    # branch_loading_file = os.path.join(base_dir, f'branch_loading_{num_cases}_cases.npz')
    # u_data = np.load(branch_loading_file)['branch_loading']
    # u_data2 = u_data[case_number].reshape(1, 1, u_data.shape[-1])

    # Q, Q2, MLSS = 120.1349, 87.5663, 4.7475
    # V0, k = 15.9184, 72.2825
    # V0, r_h_a, r_p_a1 = 15.9184, 500, 5000
    
    V0 = settling_params['v0']  # Max settling velocity (m/day)

    # only for vesilian model
    k = settling_params['Kv']           # Vesilind settling parameter (m^3/g)
    r_h_a = settling_params['r_h']      # Takacs settling parameter (m^3/g)
    r_p_a1 = settling_params['r_p']    # Takacs settling parameter (m^3/g)
    f_ns = settling_params['f_ns']    # Non-settleable solids fraction (0-1)

    V0_unit_scale = 1/24 # m3/d (bsm1) to m3/h (NN)
    k_settling_unit_scale = 1450*1000/math.log(10) # only for vesilian model # sludge density * unit scaling. k = a*ln(10)/desnity. the NN takes a as input not k so a = k*density/ln(10)*unit conversion. 
    a_a1_settling_unit_scale = (1-f_ns)*1450*1000 # only for takacs model. r_h_a = a = (1 - f_ns) * rho_d * r_h.
    # I call the input to the NN k bt it is actually a
    V0_scaled = V0 * V0_unit_scale
    # k_scaled = k * settling_unit_scale        
    r_h_a_scaled = r_h_a * a_a1_settling_unit_scale
    r_p_a1_scaled = r_p_a1 * a_a1_settling_unit_scale

    H4 = 21.9 # clarifier radius. calcaulted from bsm1. Area = = pi (rout^2-rin^2). A=1500, r_in = 0.5m, 
    Y3 = 4 #tank side depth. calcaulted from bsm1. In clarifier nomenclature, side‑water depth is the vertical water depth at the wall

    H1 = 0.5 # inlet pipe radius. h1 assumed from the range to be 0.5.
    H2 = 0.2 * H4 # hopper radius. H2 assumed from the range to be 15% tank radius. to be less than h3. 
    H3 = 0.25 * H4 # feedwell radius. h3 assumed from the range to be 20% tank radius h4.
    Y2 = 0.4 * Y3 # feedwell depth. Y2 assumed from the range to be 40% tank side depth. 30–50% of the side-water depth
    Y4 = 1.5 # hopper depth. Y4 assumed from the range to be 0.5.
    Theta1 = 5 # Floor slope indegrees. assumed from the common ranges

    # print(f"H1: {H1}")
    # print(f"H2: {H2}")
    # print(f"H3: {H3}")
    # print(f"H4: {H4}")
    # print(f"Y2: {Y2}")
    # print(f"Y3: {Y3}")
    # print(f"Y4: {Y4}")
    # print(f"Theta1: {Theta1}")
    # print(f"Q_scaled: {Q_scaled}")
    # print(f"Q2_scaled: {Q2_scaled}")
    # print(f"MLSS_scaled: {MLSS_scaled}")
    # print(f"V0_scaled: {V0_scaled}")
    # print(f"r_h_a_scaled: {r_h_a_scaled}")
    # print(f"r_p_a1_scaled: {r_p_a1_scaled}")

    # u_data2 = np.array([H1, H2, H3, H4, Y2, Y3, Y4, Theta1, Q_scaled, Q2_scaled, MLSS_scaled, V0_scaled, k_scaled]).reshape(1, 1, 13)
    u_data2 = np.array([H1, H2, H3, H4, Y2, Y3, Y4, Theta1, Q_scaled, Q2_scaled, MLSS_scaled, V0_scaled, r_h_a_scaled, r_p_a1_scaled]).reshape(1, 1, 14)
    # print(f"u_data2: {u_data2.shape}")

    # trunk_coordinates_file = os.path.join(base_dir, f'trunk_coordinates_{num_cases}_cases_{num_points}_points.npz')
    # y_data_coordinates = np.load(trunk_coordinates_file)['trunk_coordinates']
    # y_data_coordinates2 = y_data_coordinates[case_number].reshape(1, -1, 2)

    # Calculate concentrations
    ess_avg, ras_avg, prediction_data = calculate_concentrations(u_data2, nn_model, nn_u_scaler, nn_y_scaler, nn_s_scaler, nn_y_indices)

    # Print results
    # print_results(ess_avg, ras_avg, prediction_data)

    # Optional: Plot contour (comment out if not needed)
    if plot:
        update_blockMesh_from_source(H1, H2, H3, H4, Y2, Y3, Y4, Theta1)
        run_blockmesh_output()
        y_data_coordinates2 = extract_trunk_spatial_fixed().reshape(1, -1, 2)
        # print(f"Coordinates shape: {y_data_coordinates2.shape}")
        prediction_data['y_data_coordinates'] = y_data_coordinates2
        plot_contour(prediction_data)

    return ess_avg, ras_avg

def run_prediction_pipeline_separate(plot=False):
    """
    Runs the full pipeline:
    - (Optionally) load branch loading and coordinates (kept as comments per original)
    - Build u_data2 from given scalars
    - Update and run blockMesh
    - Extract trunk spatial coordinates
    - Calculate concentrations
    - Print results
    - (Optionally) plot contour

    Returns:
        ess_avg, ras_avg
    """

    # base_dir = '/lustre/isaac24/scratch/mshatara/openfoam/mshatarah-v2306/run/Secondary_Clarifier/1000_cases/Data_extracted/concatenated_cases'
    # num_cases = 703
    # num_points = 3780
    # case_number = 1
    # branch_loading_file = os.path.join(base_dir, f'branch_loading_{num_cases}_cases.npz')
    # u_data = np.load(branch_loading_file)['branch_loading']
    # u_data2 = u_data[case_number].reshape(1, 1, u_data.shape[-1])

    Q, Q2, MLSS = 1000, 768, 4.7475
    # V0, k = 15.9184, 72.2825
    V0, r_h_a, r_p_a1 = 15.9184, 500, 5000

    # I call the input to the NN k bt it is actually a

    H4 = 21.9 # clarifier radius. calcaulted from bsm1. Area = = pi (rout^2-rin^2). A=1500, r_in = 0.5m, 
    Y3 = 4 #tank side depth. calcaulted from bsm1. In clarifier nomenclature, side‑water depth is the vertical water depth at the wall

    H1 = 0.5 # inlet pipe radius. h1 assumed from the range to be 0.5.
    H2 = 0.2 * H4 # hopper radius. H2 assumed from the range to be 15% tank radius. to be less than h3. 
    H3 = 0.25 * H4 # feedwell radius. h3 assumed from the range to be 20% tank radius h4.
    Y2 = 0.4 * Y3 # feedwell depth. Y2 assumed from the range to be 40% tank side depth. 30–50% of the side-water depth
    Y4 = 1.5 # hopper depth. Y4 assumed from the range to be 0.5.
    Theta1 = 5 # Floor slope indegrees. assumed from the common ranges

    # u_data2 = np.array([H1, H2, H3, H4, Y2, Y3, Y4, Theta1, Q_scaled, Q2_scaled, MLSS_scaled, V0_scaled, k_scaled]).reshape(1, 1, 13)
    u_data2 = np.array([H1, H2, H3, H4, Y2, Y3, Y4, Theta1, Q, Q2, MLSS, V0, r_h_a, r_p_a1]).reshape(1, 1, 14)
    print(f"u_data2: {u_data2.shape}")

    nn_model = load_model()
    nn_u_scaler, nn_y_scaler, nn_s_scaler = load_scalers()
    nn_y_indices = np.load('NN_files/trunk_indices_3780_points.npz')['trunk_indices']

    # Calculate concentrations
    ess_avg, ras_avg, prediction_data = calculate_concentrations(u_data2, nn_model, nn_u_scaler, nn_y_scaler, nn_s_scaler, nn_y_indices)

    # Print results
    print_results(ess_avg, ras_avg, prediction_data)

    # Optional: Plot contour (comment out if not needed)
    if plot:
        update_blockMesh_from_source(H1, H2, H3, H4, Y2, Y3, Y4, Theta1)
        run_blockmesh_output()
        y_data_coordinates2 = extract_trunk_spatial_fixed().reshape(1, -1, 2)
        print(f"Coordinates shape: {y_data_coordinates2.shape}")
        prediction_data['y_data_coordinates'] = y_data_coordinates2
        plot_contour(prediction_data)

    return ess_avg, ras_avg

# if __name__ == '__main__':
#     run_prediction_pipeline_separate(plot=True)