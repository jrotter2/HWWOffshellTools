import glob
import json

sample_masses = [125,130, 140, 150, 160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]

#base_file_path = "/eos/user/t/tcarnaha/Summer_2022/HWW_Ntuples/Summer20UL16_106x_nAODv9_noHIPM_Full2016v9/AddLHE_MEs__AddMC_baseW/"
#base_file_path = "/eos/user/t/tcarnaha/Summer_2022/HWW_Ntuples/Summer20UL16_106x_nAODv9_HIPM_Full2016v9/AddLHE_MEs__AddMC_baseW/"
#base_file_path = "/eos/user/t/tcarnaha/Summer_2022/HWW_Ntuples/Summer20UL17_106x_nAODv9_Full2017v9/AddLHE_MEs__AddMC_baseW/"
#base_file_path = "/eos/user/t/tcarnaha/Summer_2022/HWW_Ntuples/Summer20UL18_106x_nAODv9_Full2018v9/AddLHE_MEs__AddMC_baseW/"

#base_file_path = "/eos/user/t/tcarnaha/Summer_2022/HWW_Ntuples/Autumn18_102X_nAODv7_Full2018v7/MCl1loose2018v7__MCCorr2018v7__l2loose__l2tightOR2018v7__AddLHE_MEs/"
#base_file_path = "/eos/user/t/tcarnaha/Summer_2022/HWW_Ntuples/Autumn18_102X_nAODv7_Full2018v7/AddLHE_MEs__AddMC_baseW__AddHWW_Offshell_Wgts_v2/"
#base_file_path = "/eos/user/t/tcarnaha/Summer_2022/HWW_Ntuples/Autumn18_102X_nAODv7_Full2018v7/AddLHE_MEs__AddMC_baseW__AddHWW_Offshell_Wgts_v3/"
base_file_path = "/eos/user/t/tcarnaha/Summer_2022/HWW_Ntuples/Autumn18_102X_nAODv7_Full2018v7/AddLHE_MEs__AddMC_baseW__AddHWW_Offshell_Wgts_v3__MCl1loose2018v7__MCCorr2018v7__l2loose__l2tightOR2018v7/"

base_file_path = "/eos/user/t/tcarnaha/Summer_2022/HWW_Ntuples/Summer20UL18_106x_nAODv9_Full2018v9/AddLHE_MEs__AddMC_baseW__AddHWW_Offshell_Wgts/"

file_name_match = "nanoLatino_VBFHToWWTo2L2Nu_M"
#file_name_match = "nanoLatino_GluGluHToWWTo2L2Nu_M"

#output_json_name="dataset_config_Summer20UL16_106x_nAODv9_noHIPM_Full2016v9_VBF.json"
#output_json_name="dataset_config_Summer20UL16_106x_nAODv9_HIPM_Full2016v9_VBF.json"
#output_json_name="dataset_config_Summer20UL17_106x_nAODv9_Full2017v9_VBF.json"
#output_json_name="dataset_config_Summer20UL18_106x_nAODv9_Full2018v9_VBF.json"

output_json_name = "VBF_Summer20UL18_OffshellWgtsIncluded.json"

#JSON_sample_tag = "HM_VBF_new_"
JSON_sample_tag = "HM_VBF_18_"


output_data = {}
for sample_mass in sample_masses:
    full_name = base_file_path + file_name_match + str(sample_mass) +"_*.root";
    print(full_name)
    sample = {}
    sample["files"] = glob.glob(full_name)
    if(len(sample["files"]) == 0):
        print("Missing Files for sample mass: " + str(sample_mass))
    output_data[JSON_sample_tag + str(sample_mass)] = sample

output_json = json.dumps(output_data, indent=4)
with open(JSON_sample_tag + "" + output_json_name, "w") as outfile:
    outfile.write(output_json)

