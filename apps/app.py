import pandas as pd
import numpy as np
import numbers
import random
import functools
import json

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

# |===================================|



class App:
    def __init__(self, name: str, pred_col, test_file_path: str):
        self.name = name
        self.test_file_path = test_file_path
        self.pred_col = pred_col
        self.df = None

        assert app_params[name] != None, "app does not exist in app_params"
        self.default_params = app_params[name]["default"]
        self.range_params = app_params[name]["range"]

    # returns empty string if not no input file exists
    def make_input_file(self, file_dir: str):
        return ""

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
        assert 1==0, "MAKE_FILE NOT MADE..."
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
        ontents += "# " + params_to_string(params) + "\n\n"
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




