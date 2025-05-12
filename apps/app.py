import pandas as pd
import numpy as np

class App:
    def __init__(self, name: str, pred_col, test_file_path: str):
        self.name = name
        self.test_file_path = test_file_path
        self.pred_col = pred_col
        self.df = None

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
        super().__init__("Nekbone",pred_col,test_file_path)

    def parse(self):
        X,y = super().parse()
        PREDICTION = self.pred_col

        if PREDICTION == "timeTaken":
            y = X["timeTaken"].astype(float)
            y = y.fillna(86400.0*2)

        X = X.drop(columns="timeTaken")
        X = X.drop(columns="testNum")
        return X,y

class ExaMiniMD(App):
    def __init__(self,pred_col,test_file_path: str):
        super().__init__("ExaMiniMD",pred_col,test_file_path)

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

class LAMMPS(App):
    def __init__(self,pred_col,test_file_path: str):
        super().__init__("LAMMPS",pred_col,test_file_path)

    range_params = {}
    def parse(self):
        X,y = super().parse()
        PREDICTION = self.pred_col

        if PREDICTION == "timeTaken":
            y = X["timeTaken"].astype(float)
            y = y.fillna(86400.0*2)

        assert True==False, "NOT RANGE_PARAMS NOT IMPLEMENTED"
        for column in list(X.columns):
            if column not in self.range_params["LAMMPS"]:
                X = X.drop(columns=column)


        return X,y
     

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

class HACC_IO(App):
    def __init__(self,pred_col,test_file_path: str):
        super().__init__("HACC_IO",pred_col,test_file_path)

    def parse(self):
        X,y = super().parse()
        PREDICTION = self.pred_col

        if PREDICTION == "timeTaken":
            y = X["timeTaken"].astype(float)
            y = y.fillna(86400.0*2)

        return X,y




