# This code finds a point at the top and assign it as a ref cell with Atmospheric pressure because we are using symmetry bc at the top.
#!/usr/bin/env python3
import os
import subprocess
import re
import sys

def find_max_value_position(filepath):
    """Return the 1-based index (within the data block) of the max value in the given file."""
    with open(filepath, 'r') as f:
        lines = f.readlines()

    # The number of cells is at line 22 (1-based) => lines[21] in 0-based
    N = int(lines[21].strip())

    # The data starts at line 24 (1-based) => lines[23] in 0-based
    data_lines = lines[23 : 23 + N]

    values = [float(dl.strip()) for dl in data_lines]
    max_value = max(values)
    max_index = values.index(max_value)

    # Adjust for 1-based indexing
    row_in_block = (24 + max_index) - 23

    print("Number of cells (N):", N)
    print("Maximum value found:", max_value)
    print("Its position within the data block:", row_in_block)
    return row_in_block

def update_pRefCell(location, cell_index):
    """Updates 'pRefCell' in fvSolution from 0 to the new cell index."""
    pattern = re.compile(r'^(\s*)pRefCell\s+(\d+);')
    with open(location, 'r') as f:
        lines = f.readlines()

    updated_lines = []
    for line in lines:
        match = pattern.match(line)
        if match:
            indentation = match.group(1)
            new_line = f"{indentation}pRefCell  {cell_index};\n"
            updated_lines.append(new_line)
        else:
            updated_lines.append(line)

    with open(location, 'w') as f:
        f.writelines(updated_lines)

def auto_process(case_dir):
    """
    Perform the entire process:
    1) cd to case_dir
    2) Run 'postProcess -func writeCellCentres -time latestTime'
    3) Find max value's index in 'constant/Cy'
    4) Update 'pRefCell' in 'system/fvSolution'
    """
    if not case_dir:
        print("Error: No case directory provided.")
        print("Usage: auto_process('/path/to/myOpenFOAMCase')")
        return

    # 1) Change to the specified directory
    try:
        os.chdir(case_dir)
    except FileNotFoundError:
        print(f"Error: Could not cd to '{case_dir}'.")
        return

    # 2) Run postProcess
    print(f"Running postProcess in '{case_dir}'...")
    subprocess.run(["postProcess", "-func", "writeCellCentres", "-time", "latestTime"], check=True)

    # 3) Find max value's position in constant/Cy
    cy_file_path = os.path.join(case_dir, "constant", "Cy")
    row_in_block = find_max_value_position(cy_file_path)

    # 4) Update pRefCell in system/fvSolution
    fv_solution_path = os.path.join(case_dir, "system", "fvSolution")
    update_pRefCell(fv_solution_path, row_in_block)

if __name__ == "__main__":
    # If you want the script to always use the current directory as the case_dir:
    case_dir = os.getcwd()
    # Alternatively, you could accept an argument from the user:
    #   case_dir = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    auto_process(case_dir)

