import pandas as pd
import numpy as np
import os
import copy
import numbers
import random
import platform
import functools
import json

from . import consts

from pathlib import Path


TEST_DIR = "./tests/"
DEBUG_APPS = False
SYSTEM = platform.node()

# ExaMiniMDbase

app_params = json.load(open('./apps/params.json'))

# |===== helpers with params stuff =====|
def get_pow_2(upper,lower=1):
    """ Get a uniform random power of 2 between 1 and limit.
    """
    max_bits = upper.bit_length() - 1
    min_bits = lower.bit_length() - 1
    power = random.randint(min_bits, max_bits)
    return 2 ** power
    
def factors(n):
    return set(functools.reduce(list.__add__, ([i, n//i] for i in range(1, int(n**0.5) + 1) if n % i == 0)))

def fill_empty_params(params, range_params):
    # Explicitly fill in unused parameters with None.
    # This is important to ensure default values aren't used,
    # to ensure CSV alignment, and to have some default value to train on.
    for param, values in range_params.items():
        if param not in params:
            params[param] = None

    return params 

def params_to_string(params):
    """ Convert the parameters list to a string.
    Used as comments on input files to make the parameters used clear.
    """
    string = ""
    for param in params:
        string += param + "=" \
            + str("None" if params[param] is None else params[param]) + ","
    return string

#[TODO] clean this up later once we get more info...
def get_command(app, params):
    """ Build up the appropriate command for this application run.
    """
    # Get the executable.
    if "voltrino" in SYSTEM:
        exe = "/projects/ovis/UCF/voltrino_run/"
    elif "eclipse" in SYSTEM or "ghost" in SYSTEM:
            exe = "/projects/ovis/UCF/eclipse_run/"
    else:
        exe = "../../../"

    if app.startswith("ExaMiniMD"):
        exe += "ExaMiniMD" + "/" + "ExaMiniMD"
    elif app == "HACC-IO":
        exe += "HACC-IO" + "/" + "hacc_io"
    elif app.startswith("LAMMPS"):
        exe += "LAMMPS" + "/"
        if "voltrino" in SYSTEM:
            exe += "lmp_voltrino"
        elif "eclipse" in SYSTEM or "ghost" in SYSTEM:
            exe += "LAMMPS"
    elif "nekbone" in app:
        exe = "/projects/ovis/UCF/eclipse_apps/nekbone-3.1/test/example1/nekbone"
    else:
        exe += str(app) + "/" + str(app)
    # nekbone, LAMMPS and HACC-IO don't have debug builds.
    if DEBUG_APPS and app != "nekbone" and app != "LAMMPS" and app != "HACC-IO":
        exe += ".g"

    args = ""
    if app.startswith("ExaMiniMD"):
        args = "-il input.lj"
        args += " --comm-type MPI --kokkos-threads={tasks}".format_map(params)
    elif app.startswith("LAMMPS"):
        args = "-i in.ar.lj"
    elif app == "SWFFT":
        # Locally adjust the params list to properly handle None.
        for param in params:
            params[param] = params[param] is not None and params[param] or ''
        args = "{n_repetitions} {ngx} {ngy} {ngz}".format_map(params)
    elif app == "sw4lite":
        args = "input.in"
    elif app == "nekbone":
        args = ""
    elif app == "miniAMR":
        for param in params:
            # Each of our standard parameters starts with "--".
            if param.startswith("--"):
                # If the parameter is unset, don't add it to the args list.
                if not params[param] or params[param] is None or params[param] == '':
                    continue
                # If the parameter is a flag with no value, add it alone.
                if params[param] is True:
                    args += param + " "
                # Standard parameters add their name and value.
                else:
                    args += param + " " + str(params[param]) + " "
            # load is a special case.
            # Its value is the argument.
            if param == "load":
                args += "--" + params["load"] + " "
        # Create the number of objects we need to specify.
        for _ in range(params["--num_objects"]):
            # Fill in each of these arguments.
            args += "--object {type} {bounce} {center_x} {center_y} {center_z} \
                {movement_x} {movement_y} {movement_z} {size_x} {size_y} \
                {size_z} {inc_x} {inc_y} {inc_z} ".format_map(params)
    elif app == "HACC-IO":
        args = "{numParticles} ".format_map(
            params)
        args += "/projects/ovis/UCF/data/" + SYSTEM + str(params["testNum"])

    # Assemble the whole command.
    command = exe + " " + args
    return command

# |===================================|



class App:
    def __init__(self, name: str, pred_col, test_file_path: str):
        self.name = name
        self.test_file_path = test_file_path
        self.pred_col = pred_col
        self.df = None
        self.features = {}

        assert app_params[name] != None, "app does not exist in app_params"
        self.default_params = app_params[name]["default"]
        self.range_params = app_params[name]["range"]

    def rand_param(self,param, values=''):
        """ Pick a random parameter value within a valid range.
        The approach used depends on the data type of the values."""
        if values == '':
            values = self.range_params[param]
        # If it is a boolean
        if isinstance(values[-1], bool):
            # Pick one of the values at random.
            return random.choice(values)
        # If it is a number:
        elif isinstance(values[-1], numbers.Number):
            # Get the lowest value.
            min_v = min(x for x in values if x is not None)
            # Get the highest value.
            max_v = max(x for x in values if x is not None)
            # Pick a random number between min and max to use as the parameter value.
            if isinstance(values[-1], float):
                return random.uniform(min_v, max_v)
            elif isinstance(values[-1], int):
                return random.randint(min_v, max_v)
            else:
                print("Found a range with type" + str(type(values[-1])))
                return random.randrange(min_v, max_v)
        # Else if it has no meaningful range (ex. str):
        else:
            # Pick one of the values at random.
            return random.choice(values)
    
    def make_file(self,params) -> str:
        return ""
 

    def get_params(self):   
        print(f"getting {self.name} params")
        params = {}
        params["nodes"] = get_pow_2(max(x for x in self.range_params["nodes"] if x is not None))
        params["tasks"] = get_pow_2(max(x for x in self.range_params["tasks"] if x is not None))

        # default case
        for param, values in self.range_params.items():
            if param not in params:
                params[param] = self.rand_param(param)
        
        
        return fill_empty_params(params,self.range_params) 


    def parse(self):
        """
        parses test_file.csv into a dataframe and then returns X,y
        """
        self.df = pd.read_csv(self.test_file_path,
                              sep=",", header=0, index_col=0,
                              engine="c", quotechar="\"")
        y = self.df[self.pred_col]
        X = self.df

        #[NOTE]: this is useful for error handling for the datasets
        #        if we want more apps that won't use this functionality we can abstract it later...
        def hasError(error):
            if not isinstance(error, str):
                return False
            lines = error.split('\n')
            for line in lines:
                if "Plugin file not found" in line:
                    continue
                if "error" in line or "fatal" in line or "libhugetlbfs" in line:
                    return True
            return False

        # REMOVE_ERRORS = True

        if "error" in X.columns:
            # if REMOVE_ERRORS:
            X['error'] = X.apply(
                lambda row: hasError(row["error"]), axis=1)
            X = X.drop(columns="error")
            if X.shape[0] < 1:
                # this shouldn't happen, but idk so we will assert
                assert not True, "All tests contained errors. Skipping..."
            # else:
                # X["error"] = X["error"].notnull()


        # replacements for base cases - can abstract later for more control if we need it
        X = X.replace('on', '1', regex=True)
        X = X.replace('off', '0', regex=True)
        X = X.replace('true', '1', regex=True)
        X = X.replace('false', '0', regex=True)
        X = X.replace('.true.', '1', regex=True)
        X = X.replace('.false.', '0', regex=True)

        # assert not (REMOVE_ERRORS and self.pred_col == "error") 
        return X, y

    def generate_test(self, prod, index):
        """ Create a test instance and queue up a job to run it.
        """
        app = self.name
        # These are the defaults right now.
        script_params = {"app": app,
                        "nodes": prod["nodes"],
                        "tasks": prod["tasks"]}

        # Get the default parameters, which we will adjust.
        params = copy.copy(self.default_params)
        # Update the params based on our cartesian product.
        params.update(prod)
        # Add the test number to the list of params.
        params["testNum"] = index

        # Initialize the app's dictionary.
        if app not in self.features:
            self.features = {}
        # Add the parameters to a DataFrame.
        self.features[index] = params

        # Create a folder to hold a SLURM script and any input files needed.
        test_path = Path(TEST_DIR + app + "/" + str(index).zfill(10))
        test_path.mkdir(parents=True, exist_ok=True)

        # NOTE: Add support for compiling per-test binaries from here, if needed.

        # Generate the input file contents.
        file_string = self.make_file(params)
        # If a file_string was generated
        if file_string != "":
            # Save the contents to an appropriately named file.
            if app.startswith("ExaMiniMD"):
                file_name = "input.lj"
            elif app.startswith("LAMMPS"):
                file_name = "in.ar.lj"
            elif app == "sw4lite":
                file_name = "input.in"
            elif app == "nekbone":
                file_name = "data.rea"
            print(file_string)
            with open(test_path / file_name, "w+", encoding="utf-8") as text_file:
                text_file.write(file_string)

        if app.startswith("ExaMiniMD") and params["force_type"] == "snap":
            # Copy in Ta06A.snap, Ta06A.snapcoeff, and Ta06A.snapparam.
            with open(test_path / "Ta06A.snap", "w+", encoding="utf-8") as text_file:
                text_file.write(consts.SNAP_FILE)
            with open(test_path / "Ta06A.snapcoeff", "w+", encoding="utf-8") as text_file:
                text_file.write(consts.SNAPCOEFF_FILE)
            with open(test_path / "Ta06A.snapparam", "w+", encoding="utf-8") as text_file:
                text_file.write(consts.SNAPPARAM_FILE)

        # Get the full command, with executable and arguments.
        command = get_command(app, params)
        # Set the command in the parameters.
        # Everything else was set earlier.
        script_params["command"] = command

        if "voltrino" in SYSTEM or "eclipse" in SYSTEM or "ghost" in SYSTEM:
            # Generate the SLURM script contents.
            assert 0==1, "make slurm script is not a thing yet..."
            slurm_string = make_slurm_script(script_params)
            # Save the contents to an appropriately named file.
            with open(test_path / "submit.slurm", "w+", encoding="utf-8") as text_file:
                text_file.write(slurm_string)

        return (command,test_path)

    def scrape_output(self, output, index):
        """ Scrape the output for runtime, errors, etc.
        """
        lines = output.split('\n')
        for line in lines:
            if "error" in line or "fatal" in line or "libhugetlbfs" in line:
                if "Plugin file not found" in line:
                    continue
                if "Stale file handle" in line:
                    continue
                if "error" not in self.features[index].keys():
                    self.features[index]["error"] = ""
                self.features[index]["error"] += line + "\n"
            if line.startswith("timeTaken = "):
                self.features[index]["timeTaken"] = \
                    int(line[len("timeTaken = "):])
        return self.features
    
    def append_test(self,test):
        """ Append the test results to a CSV file for later analysis.
        """
        app = self.name

        # The location of the output CSV.
        output_file = TEST_DIR + app + "dataset.csv"
        # We only add the header if we are creating the file fresh.
        needs_header = not os.path.exists(output_file)
        # Make sure the error key is there. Otherwise, it'll be missing sometimes.
        # NOTE: We need to do this for any field that is not guaranteed to be
        # present in all tests. Additionally, things will break badly if new
        # features are added in the future. We must start from scratch in such a case.
        if "error" not in self.features[test].keys():
            self.features[test]["error"] = ""
        # Convert the test to a DataFrame.
        dataframe = pd.DataFrame(self.features[test], index=[
                                self.features[test]["testNum"]])
        # Append to CSV.
        dataframe.to_csv(output_file, mode='a', header=needs_header)

class Nekbone(App):
    def __init__(self,pred_col,test_file_path: str):
        super().__init__("nekbone",pred_col,test_file_path)

    def parse(self):
        X,y = super().parse()
        PREDICTION = self.pred_col

        if PREDICTION == "timeTaken":
            y = X["timeTaken"].astype(float)
            y = y.fillna(86400.0*2)

        X = X.drop(columns="timeTaken")
        X = X.drop(columns="testNum")
        return X,y
    
    def get_params(self):
        params = {} 

        params["nodes"] = get_pow_2(max(x for x in self.range_params["nodes"] if x is not None))
        params["tasks"] = get_pow_2(max(x for x in self.range_params["tasks"] if x is not None))

        LELT = 1000
        params["ielN"] = super().rand_param("ielN")

        assert(params["ielN"] <= LELT)
        params["iel0"] = super().rand_param("iel0", [
            min(x for x in self.range_params["iel0"] if x is not None),
            params["ielN"]])
        assert(params["iel0"] <= params["ielN"])

        params["istep"] = super().rand_param("istep")

        LX1 = 12
        params["nxN"] = super().rand_param("nxN")
        assert(params["nxN"] <= LX1)

        params["nx0"] = self.rand_param("nx0", [
            min(x for x in self.range_params["nx0"] if x is not None),
            params["nxN"]])

        assert(params["nx0"] <= params["nxN"])

        params["nstep"] = self.rand_param("nstep")

        if random.choice(range(2)):
            # Automatically find the decomposition.
            params["npx"] = 0
            params["npy"] = 0
            params["npz"] = 0
            params["mx"] = 0
            params["my"] = 0
            params["mz"] = 0
        else:
            LP = 1000
            assert(params["nodes"] <= LP)
            processes_count = params["nodes"] # TODO: Verify this is the correct number.
            proc_count = []
            proc_count.append(int(random.choice(list(factors(processes_count)))))
            proc_count.append(
                int(random.choice(list(factors(processes_count/proc_count[0])))))
            proc_count.append(int(processes_count/proc_count[0]/proc_count[1]))
            random.shuffle(proc_count)
            params["npx"] = proc_count.pop()
            params["npy"] = proc_count.pop()
            params["npz"] = proc_count.pop()

            LELT = 1000
            assert(params["tasks"] <= LELT)

            processes_count = params["tasks"] # TODO: Verify this is the correct number.
            proc_count = []
            proc_count.append(int(random.choice(list(factors(processes_count)))))
            proc_count.append(
                int(random.choice(list(factors(processes_count/proc_count[0])))))
            proc_count.append(int(processes_count/proc_count[0]/proc_count[1]))
            random.shuffle(proc_count)
            params["mx"] = proc_count.pop()
            params["my"] = proc_count.pop()
            params["mz"] = proc_count.pop()

        # Set remaining parameters.
        for param, values in self.range_params.items():
            if param not in params:
                params[param] = self.rand_param(param)
        
        return fill_empty_params(params,self.range_params) 

    def make_file(self,params) -> str:
        contents = ('{ifbrick} = ifbrick ! brick or linear geometry\n'
                    '{iel0} {ielN} {istep} = iel0,ielN(per proc),stride ! range of number of elements per proc.\n'
                    '{nx0} {nxN} {nstep} = nx0,nxN,stride ! poly. order range for nx1\n'
                    '{npx} {npy} {npz} = npx,npy,npz ! np distrb, if != np, nekbone handle\n'
                    '{mx} {my} {mz} = mx,my,mz ! nelt distrb, if != nelt, nekbone handle\n').format_map(params)
        return contents



class ExaMiniMD(App):
    def __init__(self,pred_col,test_file_path: str):
        super().__init__("ExaMiniMDbase",pred_col,test_file_path)

    def parse(self):
        X,y = super().parse()
        PREDICTION = self.pred_col

        if PREDICTION == "timeTaken":
            y = X["timeTaken"].astype(float)
            y = y.fillna(86400.0*2)

        drop_columns = [ 
            "timeTaken",
            "testNum",
            "units",
            "lattice",
            "lattice_constant",
            "lattice_offset_x",
            "lattice_offset_y",
            "lattice_offset_z",
            "lattice_ny",
            "lattice_nz",
            "ntypes",
            "type",
            "mass",
            "force_cutoff",
            "temperature_target",
            "temperature_seed",
            "neighbor_skin",
            "comm_exchange_rate",
            "thermo_rate",
            "comm_newton"
        ]

        for col in drop_columns:
            X = X.drop(columns=col)

        return X,y


    def make_file(self,params) -> str:
        contents = ""
        contents += "# " + params_to_string(params) + "\n\n"
        contents += "units {units}\n".format_map(params)
        contents += "atom_style atomic\n"
        if params["lattice_constant"] is not None:
            contents += "lattice {lattice} {lattice_constant}\n".format_map(
                params)
        else:
            contents += "lattice {lattice} {lattice_offset_x} {lattice_offset_y} {lattice_offset_z}\n".format_map(
                params)
        contents += "region box block 0 {lattice_nx} 0 {lattice_ny} 0 {lattice_nz}\n".format_map(
            params)
        if params["ntypes"] is not None:
            contents += "create_box {ntypes}\n".format_map(params)
        contents += "create_atoms\n"
        contents += "mass {type} {mass}\n".format_map(params)
        if params["force_type"] != "snap":
            contents += "pair_style {force_type} {force_cutoff}\n".format_map(
                params)
        else:
            contents += "pair_style {force_type}\n".format_map(params)
            contents += "pair_coeff * * Ta06A.snapcoeff Ta Ta06A.snapparam Ta\n"
        contents += "velocity all create {temperature_target} {temperature_seed}\n".format_map(
            params)
        contents += "neighbor {neighbor_skin}\n".format_map(params)
        contents += "neigh_modify every {comm_exchange_rate}\n".format_map(
            params)
        contents += "fix 1 all nve\n"
        contents += "thermo {thermo_rate}\n".format_map(params)
        contents += "timestep {dt}\n".format_map(params)
        contents += "newton {comm_newton}\n".format_map(params)
        contents += "run {nsteps}\n".format_map(params)
        return contents
    # def get_params(self):
        # return super()
        # params = {} 

        # params["nodes"] = get_pow_2(max(x for x in self.range_params["nodes"] if x is not None))
        # params["tasks"] = get_pow_2(max(x for x in self.range_params["tasks"] if x is not None))

        # return fill_empty_params(params,self.range_params) 



class LAMMPS(App):
    def __init__(self,pred_col,test_file_path: str):
        super().__init__("LAMMPS",pred_col,test_file_path)

    def parse(self):
        X,y = super().parse()
        PREDICTION = self.pred_col

        if PREDICTION == "timeTaken":
            y = X["timeTaken"].astype(float)
            y = y.fillna(86400.0*2)

        # assert True==False, "NOT RANGE_PARAMS NOT IMPLEMENTED"
        for column in list(X.columns):
            if column not in self.range_params:
                X = X.drop(columns=column)


        return X,y

    def make_file(self,params) -> str:
        contents = "" 
        contents += "# " + params_to_string(params) + "\n\n"
        contents += "units {units}\n".format_map(params)
        contents += "atom_style atomic\n"
        contents += "lattice {lattice} {lattice_constant}\n".format_map(params)
        contents += "region box block 0 {lattice_nx} 0 {lattice_ny} 0 {lattice_nz}\n".format_map(
            params)
        contents += "create_box {ntypes} box\n".format_map(params)
        contents += "create_atoms 1 box\n"
        contents += "mass {type} {mass}\n".format_map(params)
        contents += "velocity all create {temperature_target} {temperature_seed}\n".format_map(
            params)
        contents += "pair_style {force_type} {force_cutoff}\n".format_map(
            params)
        contents += "pair_coeff 1 1 1.0 1.0\n"
        contents += "neighbor {neighbor_skin} bin\n".format_map(params)
        contents += "neigh_modify delay 0 every {comm_exchange_rate} check no\n".format_map(
            params)
        contents += "fix 1 all nve\n"
        contents += "timestep {dt}\n".format_map(params)
        contents += "thermo {thermo_rate}\n".format_map(params)
        contents += "run {nsteps}\n".format_map(params)
        return contents

    # def get_params(self):
    #     params = {} 

    #     params["nodes"] = get_pow_2(max(x for x in self.range_params["nodes"] if x is not None))
    #     params["tasks"] = get_pow_2(max(x for x in self.range_params["tasks"] if x is not None))



    #     return fill_empty_params(params,self.range_params) 


     

class SWFFT(App):
    def __init__(self,pred_col,test_file_path: str):
        super().__init__("SWFFT",pred_col,test_file_path)

    def parse(self):
        X,y = super().parse()
        PREDICTION = self.pred_col

        if PREDICTION == "timeTaken":
            y = X["timeTaken"].astype(float)
            y = y.fillna(86400.0*2)

        X["ngy"] = np.where(X["ngy"].isna(), X["ngx"], X["ngy"])
        X["ngz"] = np.where(X["ngz"].isna(), X["ngy"], X["ngz"])


        return X,y

    def get_params(self):
        params = {} 

        params["nodes"] = get_pow_2(max(x for x in self.range_params["nodes"] if x is not None))
        params["tasks"] = get_pow_2(max(x for x in self.range_params["tasks"] if x is not None))

        params["n_repetitions"] = super().rand_param("n_repetitions")

        params["ngx"] = get_pow_2(max(x for x in self.range_params["ngx"] if x is not None),
                                  min(x for x in self.range_params["ngx"] if x is not None))

        if random.choice(range(2)):
            params["ngy"] = get_pow_2(max(x for x in self.range_params["ngy"] if x is not None),
                                      min(x for x in self.range_params["ngy"] if x is not None))
        else:
            params["ngy"] = None

        if params["ngy"] is not None and random.choice(range(2)):
            params["ngz"] = get_pow_2(max(x for x in self.range_params["ngz"] if x is not None),
                                      min(x for x in self.range_params["ngz"] if x is not None))
        else:
            params["ngz"] = None

        return fill_empty_params(params,self.range_params) 

class HACC_IO(App):
    def __init__(self,pred_col,test_file_path: str):
        super().__init__("HACC-IO",pred_col,test_file_path)

    def parse(self):
        X,y = super().parse()
        PREDICTION = self.pred_col

        if PREDICTION == "timeTaken":
            y = X["timeTaken"].astype(float)
            y = y.fillna(86400.0*2)

        return X,y

    # def get_params(self):
    #     params = {} 

    #     params["nodes"] = get_pow_2(max(x for x in self.range_params["nodes"] if x is not None))
    #     params["tasks"] = get_pow_2(max(x for x in self.range_params["tasks"] if x is not None))

    #     return fill_empty_params(params,self.range_params) 




