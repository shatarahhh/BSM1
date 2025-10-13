import os
import shutil
import numpy as np
base_dir = os.getcwd()  # This assumes the script is run from /lustre/isaac/scratch/mshatara/openfoam/mshatarah-v2306/run/3D_real/Parameter_study/case..
Main_path = os.path.dirname(os.path.dirname(base_dir))# /lustre/isaac/scratch/mshatara/openfoam/mshatarah-v2306/run/3D_real/
def create_3dCoordinates_array(C_path):
    coordinates = []
    with open(C_path, 'r') as file:
        lines = file.readlines()
        num_nodes = int(lines[21].strip())
        c_lines = lines[23:23 + num_nodes]
        for line in c_lines:
            coords = line.strip()[1:-1].split()
            coordinates.append([np.float32(coords[0]), np.float32(coords[1]), np.float32(coords[2])])
    coordinates_array = np.array(coordinates, dtype=np.float32)
    coordinates_array = coordinates_array[np.newaxis, np.newaxis, np.newaxis, :, :]
    return coordinates_array, num_nodes
    
def create_Time_array(time_steps):
    time_array = np.zeros((1, 1, time_steps, 1, 1), dtype=np.float32)
    for t in range(time_steps):
        time_array[0, 0, t, 0, 0] = (t + 1) * 10  # Time value as float32
    return time_array
    
def create_branch_loading_array(base_dir):
    case_dir_name = base_dir.split('/')[-1]
    parts = case_dir_name.split('_')
    case_number = parts[0][4:]  # Extract the case number from the directory name
    x1, x2, x3, x4, x5 = map(np.float32, parts[1:])
    branch_array = np.zeros((1, 1, 1, 1, 5), dtype=np.float32)
    branch_array[0, 0, 0, 0, :] = [x1, x2, x3, x4, x5]
    return case_number, branch_array
    
def create_velocity_concentration_arrays(base_dir, time_steps, num_nodes):
    sol_u = np.zeros((1, 1, time_steps, num_nodes, 1), dtype=np.float32)
    sol_v = np.zeros((1, 1, time_steps, num_nodes, 1), dtype=np.float32)
    sol_w = np.zeros((1, 1, time_steps, num_nodes, 1), dtype=np.float32)
    sol_c = np.zeros((1, 9, time_steps, num_nodes, 1), dtype=np.float32)

    for t in range(time_steps):
        time_step = (t + 1) * 10
        u_file_path = os.path.join(base_dir, str(time_step), "U")
        scalar_files = [os.path.join(base_dir, str(time_step), f"d{j}") for j in range(1, 10)]

        try:
            with open(u_file_path, 'r') as file:
                u_lines1 = file.readlines()
                u_lines2 = u_lines1[23:23 + num_nodes]
            if 'boundaryField' in u_lines1[22]:
                continue  # Skip this file if 'boundaryField' is found
            
            velocities = np.array([[np.float32(val) for val in line.strip()[1:-1].split()[:3]] for line in u_lines2])
            sol_u[0, 0, t, :, 0] = velocities[:, 0]
            sol_v[0, 0, t, :, 0] = velocities[:, 1]
            sol_w[0, 0, t, :, 0] = velocities[:, 2]

            for index, scalar_file in enumerate(scalar_files):
                with open(scalar_file, 'r') as file:
                    s_lines = file.readlines()[23:23 + num_nodes]
                scalars = np.array([np.float32(line.strip()) for line in s_lines])
                sol_c[0, index, t, :, 0] = scalars

        except FileNotFoundError as e:
            print(f"Warning: File not found {e.filename}")
            continue

    return sol_u, sol_v, sol_w, sol_c
    

C_path = os.path.join(Main_path, 'basecase', 'constant', 'C')  # Adjust the subpath according to your directory structure.

trunk_coordinates, num_nodes = create_3dCoordinates_array(C_path)
print('num_nodes = ', num_nodes)
print('coordinates_array shape = ', trunk_coordinates.shape)

time_steps = 360
trunk_time = create_Time_array(time_steps)
print('time_array shape = ', trunk_time.shape)

case_number, branch_loading = create_branch_loading_array(base_dir)
print('branch_loading shape:', branch_loading.shape)

branch_concentration = np.array([1e-6, 4.21696503e-06, 1.77827941e-05, 7.49894209e-05,
                                     3.16227766e-04, 1.33352143e-03, 5.62341325e-03,
                                     2.37137371e-02, 1.00000000e-01]).reshape(1, 9, 1, 1, 1)


sol_u, sol_v, sol_w, sol_c = create_velocity_concentration_arrays(base_dir, time_steps, num_nodes)
print('branch loading array shape:', branch_loading.shape)
print('branch concentration array shape:', branch_concentration.shape)
print('Time array shape:', trunk_time.shape)
print('Coordinates array shape:', trunk_coordinates.shape)
print('sol_u array shape:', sol_u.shape)
print('sol_v array shape:', sol_v.shape)
print('sol_w array shape:', sol_w.shape)
print('sol_c array shape:', sol_c.shape)

# Export the arrays as an NPZ file
output_file_path = os.path.join(Main_path, 'Data_extracted', 'separate_cases', f'3D_{case_number}_4Darrays_wholecontour_real.npz')
np.savez(output_file_path, branch_loading=branch_loading, branch_concentration=branch_concentration, trunk_time=trunk_time, trunk_coordinates=trunk_coordinates, sol_u=sol_u, sol_v=sol_v, sol_w=sol_w, sol_c=sol_c)

cases_moved_path = os.path.join(Main_path, 'finished_cases_andextracted')
if not os.path.exists(cases_moved_path):
    os.makedirs(cases_moved_path)
shutil.move(base_dir, cases_moved_path)
print(f'Data saved to {cases_moved_path}')

