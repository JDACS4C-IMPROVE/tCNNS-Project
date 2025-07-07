import os
import csv
from functools import reduce
import numpy as np
import numbers
import math
from pathlib import Path
import pandas as pd
import sys
from typing import Dict

# [Req] Core improvelib imports
from improvelib.applications.drug_response_prediction.config import DRPPreprocessConfig
from improvelib.utils import str2bool
import improvelib.utils as frm

# [Req] Application-specific (DRP) imports
import improvelib.applications.drug_response_prediction.drp_utils as drp

# [Req] Model-specifc imports
from model_params_def import preprocess_params

# [Req]
filepath = Path(__file__).resolve().parent

# ---------------------------------------------
# Functions related to preprocessing drug data 
# ---------------------------------------------
def string2smiles_list(string):
    char_list = []
    i = 1
    while i < len(string):
        c = string[i]
        if c.islower():
            char_list.append(string[i-1:i+1])
            i += 1
        else:
            char_list.append(string[i-1])
        i += 1
    if not string[-1].islower():
        char_list.append(string[-1])
    return char_list

def onehot_encode(char_list, smiles_string, length):
    encode_row = lambda char: map(int, [c == char for c in smiles_string])
    ans = np.array([list(x) for x in list(map(encode_row, char_list))])
    if ans.shape[1] < length:
        residual = np.zeros((len(char_list), length - ans.shape[1]), dtype=np.int8)
        ans = np.concatenate((ans, residual), axis=1)
    return ans

def smiles_to_onehot(smiles, c_chars, c_length):
    c_ndarray = np.ndarray(shape=(len(smiles), len(c_chars), c_length), dtype=np.float32)
    for i in range(len(smiles)):
        c_ndarray[i, ...] = onehot_encode(c_chars, smiles[i], c_length)
    return c_ndarray

def load_as_ndarray(data_dir, raw_data_subdir, raw_drug_features_file):
    '''function used for original data'''
    reader = csv.reader(open(os.path.join(data_dir, raw_data_subdir, raw_drug_features_file)))
    column_names = next(reader, None)
    smiles = np.array(list(reader), dtype=np.str)
    return smiles

def charsets(smiles, use_original_data=False):
    if use_original_data:
        union = lambda x, y: set(x) | set(y)
        c_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 2]))))
    else:
        union = lambda x, y: set(x) | set(y)
        c_chars = list(reduce(union, map(string2smiles_list, list(smiles[:, 1]))))
    return c_chars

def smiles_chars(drug_df):
    '''function used for csa data'''
    # convert to dataframe to numpy
    drug_array = drug_df.to_numpy()
    # get list of unique characters
    c_chars = charsets(drug_array)
    # get length of longest SMILES string
    c_length = max(map(len, map(string2smiles_list, list(drug_array[:, 1]))))
    return sorted(c_chars), c_length

def save_drug_smiles_onehot(data_dir, raw_data_subdir, raw_drug_features_file, data_subdir, drug_file, use_original_data=False):
    '''function for original data'''
    smiles = load_as_ndarray(data_dir, raw_data_subdir, raw_drug_features_file)
    c_chars = charsets(smiles, use_original_data)
    c_length = max(map(len, map(string2smiles_list, list(smiles[:, 2]))))
    
    count = smiles.shape[0]
    drug_names = smiles[:, 0].astype(str)
    drug_cids = smiles[:, 1].astype(int)
    smiles = [string2smiles_list(smiles[i, 2]) for i in range(count)]
    
    canonical = smiles_to_onehot(smiles, c_chars, c_length)
    
    save_dict = {}
    save_dict["drug_names"] = drug_names
    save_dict["drug_cids"] = drug_cids
    save_dict["canonical"] = canonical
    save_dict["c_chars"] = c_chars

    np.save(os.path.join(data_dir, data_subdir, drug_file), save_dict)
    print("Saving preprocessed drug data...")
    return drug_names, drug_cids, canonical

def save_drug_smiles_onehot_csa(filepath, data_subdir, drug_df, c_chars, c_length, label="train"):
    '''function for csa data'''
    # convert to dataframe to numpy
    drug_array = drug_df.to_numpy()
    # get number of drugs
    count = drug_array.shape[0]
    # drug ids
    drug_cids = drug_array[:, 0].astype(str)
    # convert SMILES to list of characters
    smiles = [string2smiles_list(drug_array[i, 1]) for i in range(count)]
    # one hot encode SMILES
    canonical = smiles_to_onehot(smiles, c_chars, c_length)
    
    # save as dictionary
    save_dict = {}
    save_dict["drug_cids"] = drug_cids # improve_chem_id
    save_dict["canonical"] = canonical # SMILES
    save_dict["c_chars"] = c_chars # list of unique characters

    # save as npy file
    drug_file_name = f"{label}_drug_onehot_smiles.npy"
    #np.save(os.path.join(filepath, data_subdir, drug_file_name), save_dict)
    np.save(os.path.join(data_subdir, drug_file_name), save_dict)
    print("Saving preprocessed drug {} data...".format(label))
    return drug_cids

# ----------------------------------------------
# Functions related to preprocessing omics data
# ----------------------------------------------
def save_cell_mut_matrix(data_dir, raw_data_subdir, raw_genetic_features_file, data_subdir, cell_file):
    '''function for original data'''
    f = open(os.path.join(data_dir, raw_data_subdir, raw_genetic_features_file))
    reader = csv.reader(f)
    column_names = next(reader, None)
    print(column_names)
    cell_dict = {}
    mut_dict = {}
    id_dict = {}

    matrix_list = []
    organ1_dict = {}
    organ2_dict = {}
    for item in reader:
        cell = item[0]
        mut = item[5]
        cid = item[1]
        organ1_dict[cell] = item[2]
        organ2_dict[cell] = item[3]
        is_mutated = int(item[6])
        if cell in cell_dict:
            row = cell_dict[cell]
        else:
            row = len(cell_dict)
            cell_dict[cell] = row
        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col
        if cid in id_dict:
            row = id_dict[cid]
        else:
            row = len(id_dict)
            id_dict[cid] = row
        matrix_list.append((row, col, is_mutated))
    
    matrix = np.ones(shape=(len(cell_dict), len(mut_dict)), dtype=np.float32)
    matrix = matrix * -1
    for item in matrix_list:
        matrix[item[0], item[1]] = item[2]

    feature_num = [len(list(filter(lambda x: x >=0, list(matrix[i, :])))) for i in range(len(cell_dict))]
    indics = [i for i in range(len(feature_num)) if feature_num[i]==735]
    matrix = matrix[indics, :]

    inv_cell_dict = {v:k for k,v in cell_dict.items()}
    all_names = [inv_cell_dict[i] for i in range(len(inv_cell_dict))]
    cell_names = np.array([all_names[i] for i in indics])

    inv_id_dict = {v:k for k,v in id_dict.items()}
    all_ids = [inv_id_dict[i] for i in range(len(inv_id_dict))]
    cell_id = np.array([all_ids[i] for i in indics])

    inv_mut_dict = {v:k for k,v in mut_dict.items()}
    mut_names = np.array([inv_mut_dict[i] for i in range(len(inv_mut_dict))])
    
    desc1 = []
    desc2 = []
    for i in range(cell_names.shape[0]):
        desc1.append(organ1_dict[cell_names[i]])
        desc2.append(organ2_dict[cell_names[i]])
    desc1 = np.array(desc1)
    desc2 = np.array(desc2)

    save_dict = {}
    save_dict["cell_mut"] = matrix
    save_dict["cell_names"] = cell_names
    save_dict["mut_names"] = mut_names
    save_dict["desc1"] = desc1
    save_dict["desc2"] = desc2
    save_dict["cell_id"] = cell_id

    np.save(os.path.join(data_dir, data_subdir, cell_file), save_dict)
    print("Saving preprocessed omics data...")

    return matrix, cell_names, mut_names, cell_id

def save_cell_mut_matrix_csa(filepath, data_subdir, gf_df, label="train", sample_name="sample_id"):
    '''function for csa data'''
    # transform from long to wide format
    gf_df_wide = gf_df.pivot_table(index=sample_name, columns="genetic_feature", values="is_mutated").reset_index()
    # convert dataframe to array
    matrix = gf_df_wide.iloc[: , 1:].to_numpy()

    mut_names = gf_df.genetic_feature.unique().tolist() # name of GDSC genetic features
    cell_id = gf_df[sample_name].unique().tolist() # improve_sample_id

    # save as dictionary
    save_dict = {}
    save_dict["cell_mut"] = matrix
    save_dict["mut_names"] = mut_names
    save_dict["cell_id"] = cell_id

    # save as npy file
    cell_file_name = f"{label}_cell_mut_matrix.npy"
    np.save(os.path.join(data_subdir, cell_file_name), save_dict)
    print("Saving preprocessed omics {} data...".format(label))
    return cell_id

# --------------------------------------------------
# Functions related to preprocessing CSA omics data
# --------------------------------------------------
def remove_dup_cna(x):
    if (x["is_mutated"] == 1) & (np.isnan(x["temp_mutated"])):
        return 1
    elif (x["is_mutated"] == 1) & (x["temp_mutated"] == 0):
        return 1
    else:
        return 0
    
def loss_gain(x):
    if (x["value"] < 0):
        return "loss"
    elif (x["value"] > 0):
        return "gain"
    else:
        return "neutral"
    
def get_mutations(mut_df, gene_df, sample_name="sample_id"):
    # reshape from wide to long format
    mut_df = pd.melt(mut_df, id_vars=sample_name, value_vars=mut_df.columns[1:].to_list())
    # if any mutation in gene, indicate with 1
    mut_df["is_mutated"] = np.where(mut_df["value"] > 0, 1, 0)
    # drop and rename columns
    mut_df = mut_df.drop(columns=["value"]).rename(columns={"variable": "genetic_feature"})
    # list of GDSC mutation genes
    mut_list = gene_df[gene_df["Genetic Feature"].str.contains("mut")]["Genes in Segment"].tolist()
    # filter to GDSC mutation genes
    return mut_df[mut_df.genetic_feature.isin(mut_list)]

def get_copy_number_alterations(cna_df, gene_df, sample_name="sample_id"):
    # reshape from wide to long format
    cna_df = pd.melt(cna_df, id_vars=sample_name, value_vars=cna_df.columns[1:].to_list())
    # filter to GDSC CNA genetic features
    gene_df = gene_df[gene_df["Genetic Feature"].str.contains("cna")]
    # filter to GDSC genes corresponding to CNAs
    cna_df = cna_df[cna_df.variable.isin(gene_df["Genes in Segment"].tolist())]
    # categorize value as gain, loss, or neutral
    cna_df = cna_df.assign(loss_gain = cna_df.apply(loss_gain, axis=1))
    # merge dataframes to match CNA genetic feature with genes in segment
    cna_df = cna_df.merge(gene_df, left_on="variable", right_on="Genes in Segment")
    # if discretized copy number category matches with CNA loss/gain type, indicate with a 1
    cna_df["is_mutated"] = np.where(cna_df["loss_gain"] == cna_df["Recurrent Gain Loss"], 1, 0)
    # remove columns and keep only unique rows
    cna_df = cna_df.drop(columns = ["value","loss_gain","Recurrent Gain Loss","Genes in Segment","variable"]).drop_duplicates()
    # filter to CNAs with mutations
    has_mut_df = cna_df[cna_df.is_mutated == 1]
    # filter to CNAs without mutations
    no_mut_df = cna_df[cna_df.is_mutated == 0].rename(columns={"is_mutated":"temp_mutated"})
    # merge dataframes
    new_cna_df = has_mut_df.merge(no_mut_df, on=[sample_name,"Genetic Feature"], how="outer")
    # prioritize CNAs with at least one gene mutated to finalize "is_mutated"
    new_cna_df = new_cna_df.assign(final_is_mutated = new_cna_df.apply(remove_dup_cna, axis=1))
    # drop and rename columns
    return new_cna_df.drop(columns=["is_mutated","temp_mutated"]).rename(columns={"final_is_mutated":"is_mutated", "Genetic Feature":"genetic_feature"})

# --------------------------------------------------
# Functions related to preparing drug response data
# --------------------------------------------------
def norm_ic50(x):
    return 1 / (1 + pow(math.exp(float(x)), -0.1))

def save_drug_cell_matrix(data_dir, raw_data_subdir, raw_drug_features_file, raw_genetic_features_file, raw_drug_response_file, 
                          data_subdir, drug_file, cell_file, response_file):
    '''function for original data'''
    f = open(os.path.join(data_dir, raw_data_subdir, raw_drug_response_file))
    reader = csv.reader(f)
    column_names = next(reader, None)

    drug_dict = {}
    cell_dict = {}
    matrix_list = []

    for item in reader:
        drug = item[0]
        cell = item[3] # cell ids
        
        if drug in drug_dict:
            row = drug_dict[drug]
        else:
            row = len(drug_dict)
            drug_dict[drug] = row
        if cell in cell_dict:
            col = cell_dict[cell]
        else:
            col = len(cell_dict)
            cell_dict[cell] = col
        
        matrix_list.append((row, col, item[8], item[9], item[10], item[11], item[12]))
        
    existance = np.zeros(shape=(len(drug_dict), len(cell_dict)), dtype=np.int32)
    matrix = np.zeros(shape=(len(drug_dict), len(cell_dict), 6), dtype=np.float32)
    for item in matrix_list:
        existance[item[0], item[1]] = 1
        matrix[item[0], item[1], 0] = 1 / (1 + pow(math.exp(float(item[2])), -0.1))
        matrix[item[0], item[1], 1] = float(item[3])
        matrix[item[0], item[1], 2] = float(item[4])
        matrix[item[0], item[1], 3] = float(item[5])
        matrix[item[0], item[1], 4] = float(item[6])
        matrix[item[0], item[1], 5] = math.exp(float(item[2]))

    inv_drug_dict = {v:k for k,v in drug_dict.items()}
    inv_cell_dict = {v:k for k,v in cell_dict.items()}
    
    drug_names, drug_cids, canonical = save_drug_smiles_onehot(data_dir, raw_data_subdir, raw_drug_features_file, data_subdir, drug_file)
    cell_mut_matrix, cell_names, mut_names, cell_ids = save_cell_mut_matrix(data_dir, raw_data_subdir, raw_genetic_features_file, data_subdir, cell_file)
    
    d_ids = [drug_dict[i] for i in drug_names]
    c_ids = [cell_dict[i] for i in cell_ids]
    sub_matrix = matrix[d_ids, :][:, c_ids]
    existance = existance[d_ids, :][:, c_ids]
    
    row, col = np.where(existance > 0)
    positions = np.array(zip(row, col))
   
    save_dict = {}
    save_dict["drug_names"] = drug_names
    save_dict["cell_names"] = cell_names
    save_dict["cell_ids"] = cell_ids
    save_dict["positions"] = positions
    save_dict["IC50"] = sub_matrix[:, :, 0]
    save_dict["AUC"] = sub_matrix[:, :, 1]
    save_dict["Max_conc"] = sub_matrix[:, :, 2]
    save_dict["RMSE"] = sub_matrix[:, :, 3]
    save_dict["Z_score"] = sub_matrix[:, :, 4]
    save_dict["raw_ic50"] = sub_matrix[:, :, 5]

    np.save(os.path.join(data_dir, data_subdir, response_file), save_dict)
    print("Saving preprocessed response data...")
    return sub_matrix

def save_drug_cell_matrix_csa(filepath, data_subdir, rs_df, d_id, c_id, label="train", response_label="auc1", drug_col_name="drug_id", canc_col_name="sample_id"):
    '''function for csa data'''
    # when using IC50, normalize the logarithmic IC50 values in (0,1) interval as described in Liu et al
    if response_label == "ic50":
        # remove missing IC50 values
        rs_df = rs_df[~rs_df.ic50.isna()]
        rs_df[response_label] = rs_df[response_label].apply(norm_ic50)
        rs_response_vals = rs_df[response_label].tolist()
    else:
        rs_response_vals = rs_df[response_label].tolist()
    
    # get list of drug and cell IDs in response dataframe
    rs_drug_cids = rs_df[drug_col_name].tolist()
    rs_cell_ids = rs_df[canc_col_name].tolist()
    rs_temp_cell_ids = rs_df["temp_sample_id"].tolist()
    
    # save as dictionary
    save_dict = {}
    save_dict["drug_cid"] = rs_drug_cids
    save_dict["cell_ids"] = rs_cell_ids
    save_dict["temp_cell_ids"] = rs_temp_cell_ids
    save_dict["response_values"] = rs_response_vals
    
    # get indices of drug and cell IDs for SMILES and genetic features matrices
    d_dict = {val: idx + 0 for idx, val in enumerate(d_id)}
    d_index = [d_dict[i] for i in d_id]
    d_pos = [d_dict[i] for i in rs_drug_cids]
    c_dict = {val: idx + 0 for idx, val in enumerate(c_id)}
    c_pos = [c_dict[i.split("_")[0]] for i in rs_temp_cell_ids]
    t_dict = {val: idx + 0 for idx, val in enumerate(list(set(rs_temp_cell_ids)))}
    t_index = [t_dict[i] for i in list(set(rs_temp_cell_ids))]
    t_pos = [t_dict[i] for i in rs_temp_cell_ids]
    
    positions = np.array(list(np.array(zip(d_pos, c_pos, t_pos)).tolist()))
    save_dict["positions"] = positions

    # create matrix of drug response values
    matrix = np.zeros(shape=(len(d_dict), len(t_dict), 1), dtype=np.float32)
    for idx, x in enumerate(positions):
        matrix[x[0], x[2], 0] = rs_response_vals[idx]
    sub_matrix = matrix[d_index, :][:, t_index]
    save_dict[response_label] = sub_matrix
    
    # save as npy file
    response_file_name = f"{label}_drug_cell_interaction.npy"
    np.save(os.path.join(data_subdir, response_file_name), save_dict)
    print("Saving preprocessed response {} data...".format(label))


# [Req]
def run(params: Dict): 
    """ Run data preprocessing.

    Args:
        params (dict): dict of IMPROVE parameters and parsed values.

    Returns:
        str: directory name that was used to save the ML data files.
    """
    
    if params["use_original_data"]:
        # get data from server if original data is not available
        #candle.file_utils.get_file(args.original_data, f"{args.data_url}/{args.original_data}", cache_subdir = args.cache_subdir)
        # make directory for processed data 
        proc_path = os.path.join(params["data_dir"], params["data_subdir"])
        os.mkdir(proc_path)      
        save_drug_cell_matrix(params["data_dir"], params["raw_data_subdir"], params["raw_drug_features_file"], params["raw_genetic_features_file"], 
                          params["raw_drug_response_file"], params["data_subdir"], params["drug_file"], params["cell_file"], params["response_file"])
    else:
        # ------------------------------------------------------
        # [Req] Build paths and create output data dir
        # ------------------------------------------------------

        # ------------------------------------------------------
        # [Req] Load X data (feature representations)
        # ------------------------------------------------------
        print("\nLoading omics data...")
        genes_df = pd.read_csv(os.path.join(filepath, params["gdsc_gene_file"])) # load GDSC genetic features
        # Discretized copy number data
        dis_copy = frm.get_x_data(file = params['cell_cnv_file'], 
                            benchmark_dir = params['input_dir'], 
                            column_name = params['canc_col_name'])
        # Mutation count data
        mut_count = frm.get_x_data(file = params['cell_mutation_file'], 
                            benchmark_dir = params['input_dir'], 
                            column_name = params['canc_col_name'])

        print("\nLoading drugs data...")
        # Drug smiles data
        smi = frm.get_x_data(file = params['drug_smiles_file'], 
                    benchmark_dir = params['input_dir'], 
                    column_name = params['drug_col_name'])
        # ------------------------------------------------------
        # [Req] Construct ML data for every stage (train, val, test)
        # ------------------------------------------------------
        # Dict with split files corresponding to the three sets (train, val, and test)
        stages = {"train": params["train_split_file"],
                  "val": params["val_split_file"],
                  "test": params["test_split_file"]}
        
        for stage, split_file in stages.items():
            # --------------------------------
            # [Req] Load response data
            # --------------------------------
            print("\nLoading response data for {}...".format(stage))
            rsp = frm.get_y_data(split_file=split_file, 
                                benchmark_dir=params['input_dir'], 
                                y_data_file=params['y_data_file'])
            rsp = rsp.dropna(subset=[params['y_col_name']])
            # --------------------------------
            # [Req] Build data name
            # --------------------------------
            
            # --------------------------------
            # [Req] Save ML data files in params["output_dir"]
            # --------------------------------
            # Retain (cancer, drug) response pairs that have both omics and drug features available
            # (i.e., intersection of samples)
            
            # Filter responses to include only samples that have mutation count data
            ydf = frm.get_y_data_with_features(rsp, mut_count, params["canc_col_name"])
            print("Filtered response data to retain only samples with mutation count features:")
            print(ydf.shape)
            print("Unique cancer samples and drugs:", ydf[[params["canc_col_name"], params["drug_col_name"]]].nunique())

            # Further filter to retain only samples with copy number variation data
            ydf = frm.get_y_data_with_features(ydf, dis_copy, params["canc_col_name"])
            print("Further filtered to retain only samples with CNV features:")
            print(ydf.shape)
            print("Unique cancer samples and drugs:", ydf[[params["canc_col_name"], params["drug_col_name"]]].nunique())

            # Further filter to retain only samples with drug SMILES data
            ydf = frm.get_y_data_with_features(ydf, smi, params["drug_col_name"])
            print("Further filtered to retain only drugs with SMILES features:")
            print(ydf.shape)
            print("Unique cancer samples and drugs:", ydf[[params["canc_col_name"], params["drug_col_name"]]].nunique())

            # Add a temporary ID to distinguish same sample across different studies
            ydf["temp_sample_id"] = ydf[params["canc_col_name"]] + "_" + ydf["study"].astype(str)

            # Drop response pairs with missing drug response values
            ydf = ydf.dropna(subset=[params["y_col_name"]])
            print("Dropped rows with missing response values:")
            print(ydf.shape)
            print("Unique cancer samples, drugs, and temp_sample_id:", 
                ydf[[params["canc_col_name"], params["drug_col_name"], "temp_sample_id"]].nunique())
            
            # -------------------
            # Prep drug features
            # -------------------
            # Get SMILES data per drug
            #smi_subset = smi[smi[params["drug_col_name"]].isin(list(ydf[params["drug_col_name"]].unique()))]
            smi_subset = frm.get_features_in_y_data(smi, ydf, params['drug_col_name']).reset_index()
            # Get unique characters and length of longest SMILES string in entire drug dataset
            tr_chars, tr_length = smiles_chars(smi.reset_index())
            # One hot encode SMILES data and save file            
            drug_data = save_drug_smiles_onehot_csa(filepath, params["output_dir"], smi_subset, tr_chars, tr_length, label=stage)
            
            # ----------------
            # Prep omics data
            # ----------------
            # Get mutation count data in GDSC format
            mut = get_mutations(frm.get_features_in_y_data(mut_count, ydf, params['canc_col_name']).reset_index(), genes_df, params["canc_col_name"])
            # Get discreted copy number data in GDSC format
            cna = get_copy_number_alterations(frm.get_features_in_y_data(dis_copy, ydf, params['canc_col_name']).reset_index(), genes_df, params["canc_col_name"])
            # Combine and sort by sample ID and genetic feature
            gf = pd.concat([mut, cna]).sort_values([params["canc_col_name"], 'genetic_feature'])
            # Create sample and mutation matrix and save file
            omics_data = save_cell_mut_matrix_csa(filepath, params["output_dir"], gf, label=stage, sample_name=params["canc_col_name"])
            
            # -------------------
            # Prep response data
            # -------------------
            # Preprocess and save drug response file
            save_drug_cell_matrix_csa(filepath, params["output_dir"], ydf, drug_data, omics_data, label=stage, response_label=params["y_col_name"], drug_col_name = params["drug_col_name"], canc_col_name = params["canc_col_name"])
            
            # [Req] Save y dataframe for the current stage
            frm.save_stage_ydf(ydf.drop("temp_sample_id", axis=1), stage, params["output_dir"])
                   
       
    return params["output_dir"]   
 
# [Req]      
def main(args):
    cfg = DRPPreprocessConfig()
    params = cfg.initialize_parameters(
        pathToModelDir=filepath,
        default_config="tcnns_params.ini",
        additional_definitions=preprocess_params
    )
    timer_preprocess = frm.Timer()
    ml_data_outdir = run(params)
    timer_preprocess.save_timer(dir_to_save=params["output_dir"], 
                                filename='runtime_preprocess.json', 
                                extra_dict={"stage": "preprocess"})
    print("\nFinished data preprocessing.")

# [Req]
if __name__ == "__main__":
    main(sys.argv[1:])
