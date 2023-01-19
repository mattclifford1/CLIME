# author: Jonny Erskine
# email: jonathan.erskine@bristol.ac.uk

# author2: Matt Clifford
# email2: matt.clifford@bristol.ac.uk

def out(message,verbose):
    if verbose:
        print(message)

def input_error_msg(given_key, available_keys, key_name):
    error_msg = f"'{key_name}' needs to be one of: {available_keys} not: {given_key}"
    return error_msg
