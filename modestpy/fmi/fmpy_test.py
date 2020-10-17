"""Example of FMPy-based simulation."""
import json
from fmpy import simulate_fmu, dump, read_model_description, instantiate_fmu, extract
from fmpy.util import read_csv, write_csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def df_to_struct_arr(df):
    """Converts a DataFrame to structured array."""
    struct_arr = np.rec.fromrecords(df, names=df.columns.tolist())

    return struct_arr


def struct_arr_to_df(arr):
    """Converts a structured array to DataFrame."""
    df = pd.DataFrame(arr).set_index('time')

    return df


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

# Start and stop time
start_time = inp_df['time'].iloc[0]
stop_time = inp_df['time'].iloc[-1]
output_interval = inp_df['time'].iloc[1] - inp_df['time'].iloc[0]

# Reset the FMU instance instead of creating a new one
fmu.reset()

# Simulate
result = simulate_fmu(
    filename=fmu_path,
    start_values=start_values,
    start_time=start_time,
    stop_time=stop_time,
    input=inp_struct,
    output=None,
    output_interval=output_interval,
    fmu_instance=fmu
)

# Free the FMU instance and free the shared library
fmu.freeInstance()

# Result to DataFrame
result = struct_arr_to_df(result)
print(result)
plt.plot(result)
plt.show()
