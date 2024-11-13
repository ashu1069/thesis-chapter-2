from modules import *
import numpy as np
import pandas as pd


def load_static_data(file_paths):
    data_frames = [pd.read_csv(file) for file in file_paths]
    combined_df = pd.concat(data_frames, ignore_index=True)
    return combined_df

def test_grn(file_paths):
    # Load static data from CSV files
    static_data = load_static_data(file_paths)
    
    # Define the static variable list
    static_var_list = [
        'Endemic Potential R0', 'Endemic Potential CFR', 'Endemic Potential Duration',
        'Demography Urban-Rural Split', 'Demography Population Density', 'Environmental Index',
        'Communication Affordability', 'Communication Media Freedom', 'Communication Connectivity',
        'Socio-economic GDP per capita', 'Socio-economic Gini Index', 'Socio-economic Employment Rates',
        'Socio-economic Poverty Rates', 'Socio-economic Education Levels'
    ]

    # Ensure the data frame contains the required static variables
    assert all(var in static_data.columns for var in static_var_list), "Missing required static variables in the data."

    # Define input sizes for the static variables
    input_sizes = {var: 1 for var in static_var_list}

    # Create synthetic input data for the static variables
    batch_size = len(static_data)
    inputs = {var: torch.tensor(static_data[var].values).float().view(batch_size, -1) for var in static_var_list}

    # Initialize the GatedResidualNetwork for each variable
    grns = {var: GatedResidualNetwork(input_size=1, hidden_size=8, output_size=1, dropout=0.1, context_size=6) for var in static_var_list}

    # Forward pass for each variable
    outputs = {}
    context = torch.randn(batch_size, 6)  # Random context vector
    for var, grn in grns.items():
        outputs[var] = grn(inputs[var], context)

    # Print outputs
    for var, output in outputs.items():
        print(f"{var} Output: {output}")

if __name__ == '__main__':
    # List of file paths to the CSV files
    file_paths = [
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_COVID-19.csv',
    ]
    test_grn(file_paths)



def test_variable_selection_network(file_paths):
    # Load static data from CSV files
    static_data = load_static_data(file_paths)
    
    # Define the static variable list
    static_var_list = [
        'Endemic Potential R0', 'Endemic Potential CFR', 'Endemic Potential Duration',
        'Demography Urban-Rural Split', 'Demography Population Density', 'Environmental Index',
        'Communication Affordability', 'Communication Media Freedom', 'Communication Connectivity',
        'Socio-economic GDP per capita', 'Socio-economic Gini Index', 'Socio-economic Employment Rates',
        'Socio-economic Poverty Rates', 'Socio-economic Education Levels'
    ]

    # Ensure the data frame contains the required static variables
    assert all(var in static_data.columns for var in static_var_list), "Missing required static variables in the data."

    # Define input sizes for the static variables
    input_sizes = {var: 1 for var in static_var_list}

    # Create synthetic input data for the static variables
    batch_size = len(static_data)
    print(batch_size)
    inputs = {var: torch.tensor(static_data[var].values).float().view(batch_size, -1) for var in static_var_list}

    # Initialize the VariableSelectionNetwork
    vsn = VariableSelectionNetwork(
        input_sizes=input_sizes,
        hidden_size=8,
        input_embedding_flags={var: True for var in static_var_list},
        dropout=0.1,
        context_size=6
    )

    # Forward pass with a random context vector
    context = torch.randn(batch_size, 6)
    outputs, sparse_weights = vsn(inputs, context)

    # Print outputs and weights
    print("Outputs:", outputs)
    print("Sparse Weights:", sparse_weights)

if __name__ == '__main__':
    # List of file paths to the CSV files
    file_paths = [
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_Chikungunya.csv',
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_COVID-19.csv',
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_Dengue.csv',
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_Group B Streptococcus.csv',
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_Chikungunya.csv',
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_Hepatitis-E.csv',
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_M-pox.csv',
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_Shigella.csv',
        '/home/stu12/s11/ak1825/hsel/static_data/static_data_TB.csv'
    ]
    test_variable_selection_network(file_paths)