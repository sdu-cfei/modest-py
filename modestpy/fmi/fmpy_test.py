"""Exemplary use of FMPy. This script can be deleted when everything starts working."""
import json
from fmpy import simulate_fmu, dump, read_model_description, instantiate_fmu, extract
from fmpy.util import read_csv, write_csv
import numpy as np
import pandas as pd


def df_to_struct_arr(df):
    """Converts a DataFrame to a structured array."""
    struct_arr = np.rec.fromrecords(df, names=df.columns.tolist())

    return struct_arr

# Paths
fmu_path = "examples/simple/resources/Simple2R1C_ic_linux64.fmu"
input_path = "examples/simple/resources/inputs.csv"
known_path = "examples/simple/resources/known.json"
est_path = "examples/simple/resources/est.json"

# Print some info about the FMU
dump(fmu_path)

# Instantiate FMU
model_desc = read_model_description(fmu_path)
unzipdir = extract(fmu_path)
fmu = instantiate_fmu(unzipdir, model_desc)

# Input
inp_df = pd.read_csv(input_path)
inp_struct = df_to_struct_arr(inp_df)

# Parameters
with open(known_path, 'r') as f:
    start_values = json.load(f)

# Declare output names
#output = []

# Simulate using the existing FMU instance
# (for faster looping)
for i in range(3):
    # Reset the FMU instance instead of creating a new one
    fmu.reset()

    # Simulate
    result = simulate_fmu(
        filename=fmu_path,
        start_values=start_values,
        input=inp_struct,
        output=None,
        output_interval=None,
        fmu_instance=fmu
    )

# Free the FMU instance and free the shared library
fmu.freeInstance()

print(result)
