import json
import uproot
import glob

output_json_name = "test_dnn_config_v3.json"
output_data = {}


MC_BASE_PATH = "/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano/"
CUSTOM_PATH = "/eos/cms/store/group/phys_higgs/cmshww/fernanpe/HWWNano/"
YEARS = ["Summer20UL16_106x_nAODv9_HIPM_Full2016v9", "Summer20UL16_106x_nAODv9_noHIPM_Full2016v9", "Summer20UL17_106x_nAODv9_Full2017v9", "Summer20UL18_106x_nAODv9_Full2018v9"]
STEPS = ["MCl1loose2016v9__MCCorr2016v9NoJERInHorn__l2tightOR2016v9", "MCl1loose2016v9__MCCorr2016v9NoJERInHorn__l2tightOR2016v9", "MCl1loose2017v9__MCCorr2017v9NoJERInHorn__l2tightOR2017v9", "MCl1loose2018v9__MCCorr2018v9NoJERInHorn__l2tightOR2018v9"]
STEPS_CUSTOM = "AddLHE_MEs__AddMC_baseW__AddHWW_Offshell_Wgts__"


##### WW INFO #####
WW_SUBSAMPLES = ["WW" , "WWewk", "ggWW"]
WW_SUBSAMPLE_SEARCH = {"WW" : ["*_WWTo2L2Nu__*"], "WWewk" : ["*_WpWmJJ_EWK_noTop__*"], "ggWW" : ["*_GluGluToWWToENMN__*", "*_GluGluToWWToMNEN__*", "*_GluGluToWWToMNMN__*"]}
WW_SUBSAMPLE_WEIGHTS = [["XSWeight"], ["XSWeight"], ["XSWeight"]]
WW_SUBSAMPLE_CUSTOMWEIGHTS = [["SFweight","PromptGenLepMatch2l","METFilter_MC", "nllW", "ewknloW"], ["NOT_TOP", "NOT_HIGGS", "SFweight","PromptGenLepMatch2l","METFilter_MC"], ["k_factor_1p53over1p4", "SFweight","PromptGenLepMatch2l","METFilter_MC"]]

WW_SUBSAMPLE_FILES = [[], [], []]

for subsample_index, subsample in enumerate(WW_SUBSAMPLES):
    for search_q in WW_SUBSAMPLE_SEARCH[subsample]:
        for yr_index, yr in enumerate(YEARS):
            file_path = MC_BASE_PATH + yr + "/" + STEPS[yr_index] + "/" + search_q + ".root"
            files = glob.glob(file_path)
            files = [f_name.split(MC_BASE_PATH)[-1] for f_name in files]
            WW_SUBSAMPLE_FILES[subsample_index] = WW_SUBSAMPLE_FILES[subsample_index] + files

WW_subpath_dict = [{"name" : WW_SUBSAMPLES[i], "weights" : WW_SUBSAMPLE_WEIGHTS[i], "custom_weights" : WW_SUBSAMPLE_CUSTOMWEIGHTS[i], "files" : WW_SUBSAMPLE_FILES[i]} for i in range(0, len(WW_SUBSAMPLES))]


WW_sample_data = {"basepath" : "/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano/", 
                  "subpaths" : WW_subpath_dict
                 }

output_data["UL_WW"] = WW_sample_data
######################

##### top INFO ##### 
top_SUBSAMPLES = ["ST_tW_top", "ST_tW_antitop", "TTTo2L2Nu", "ST_s-channel", "ST_t-channel_top", "ST_t-channel_antitop"]
top_SUBSAMPLE_SEARCH = {"ST_tW_top" : ["*_ST_tW_top__*"], "ST_tW_antitop" :["*_ST_tW_antitop__*"], "TTTo2L2Nu" : ["*_TTTo2L2Nu__*"], "ST_s-channel" : ["*_ST_s-channel__"], "ST_t-channel_top" : ["*_ST_t-channel_top__*"], "ST_t-channel_antitop" : ["*_ST_t-channel_antitop__*"]}
top_SUBSAMPLE_WEIGHTS = [["XSWeight"], ["XSWeight"], ["XSWeight"], ["XSWeight"], ["XSWeight"], ["XSWeight"]]
top_SUBSAMPLE_CUSTOMWEIGHTS = [["SFweight","PromptGenLepMatch2l","METFilter_MC"], ["SFweight","PromptGenLepMatch2l","METFilter_MC"], ["SFweight","PromptGenLepMatch2l","METFilter_MC"], ["SFweight","PromptGenLepMatch2l","METFilter_MC"], ["SFweight","PromptGenLepMatch2l","METFilter_MC"], ["SFweight","PromptGenLepMatch2l","METFilter_MC"]]

top_SUBSAMPLE_FILES = [[], [], [], [], [], []]

for subsample_index, subsample in enumerate(top_SUBSAMPLES):
    for search_q in top_SUBSAMPLE_SEARCH[subsample]:
        for yr_index, yr in enumerate(YEARS):
            file_path = MC_BASE_PATH + yr + "/" + STEPS[yr_index] + "/" + search_q + ".root"
            print(file_path)
            files = glob.glob(file_path)
            files = [f_name.split(MC_BASE_PATH)[-1] for f_name in files]
            top_SUBSAMPLE_FILES[subsample_index] = top_SUBSAMPLE_FILES[subsample_index] + files

top_subpath_dict = [{"name" : top_SUBSAMPLES[i], "weights" : top_SUBSAMPLE_WEIGHTS[i], "custom_weights" : top_SUBSAMPLE_CUSTOMWEIGHTS[i], "files" : top_SUBSAMPLE_FILES[i]} for i in range(0, len(top_SUBSAMPLES))]

top_sample_data = {"basepath" : "/eos/cms/store/group/phys_higgs/cmshww/amassiro/HWWNano/",
                  "subpaths" : top_subpath_dict
                 }

output_data["UL_top"] = top_sample_data
######################

##### VBF INFO ##### 
mass_poles = [125,160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]
VBF_SUBSAMPLES = ["VBF_"+str(mass) for mass in mass_poles]
VBF_SUBSAMPLE_SEARCH = {}
for mass_index, mass in enumerate(mass_poles):
    VBF_SUBSAMPLE_SEARCH[VBF_SUBSAMPLES[mass_index]] = ["*_VBFHToWWTo2L2Nu_M" + str(mass) + "__*"]
VBF_SUBSAMPLE_WEIGHTS = [["XSWeight", "p_Gen_JJEW_SIG_ghv1_1_MCFM", "p_Gen_CPStoBWPropRewgt", "HWWOffshell_combineWgt"] for mass in mass_poles]
VBF_OFF_SUBSAMPLE_CUSTOMWEIGHTS = [["OFFSHELL_HIGGS"] for mass in mass_poles]
VBF_ON_SUBSAMPLE_CUSTOMWEIGHTS = [["ONSHELL_HIGGS"] for mass in mass_poles]

VBF_SUBSAMPLE_FILES = [[] for mass in mass_poles]

for subsample_index, subsample in enumerate(VBF_SUBSAMPLES):
    for search_q in VBF_SUBSAMPLE_SEARCH[subsample]:
        for yr_index, yr in enumerate(YEARS):
            file_path = CUSTOM_PATH + yr + "/" + STEPS_CUSTOM + STEPS[yr_index] + "/" + search_q + ".root"
            print(file_path)
            files = glob.glob(file_path)
            files = [f_name.split(CUSTOM_PATH)[-1] for f_name in files]
            VBF_SUBSAMPLE_FILES[subsample_index] = VBF_SUBSAMPLE_FILES[subsample_index] + files

VBF_OFF_subpath_dict = [{"name" : VBF_SUBSAMPLES[i], "weights" : VBF_SUBSAMPLE_WEIGHTS[i], "custom_weights" : VBF_OFF_SUBSAMPLE_CUSTOMWEIGHTS[i], "files" : VBF_SUBSAMPLE_FILES[i]} for i in range(0, len(VBF_SUBSAMPLES))]
VBF_ON_subpath_dict = [{"name" : VBF_SUBSAMPLES[i], "weights" : VBF_SUBSAMPLE_WEIGHTS[i], "custom_weights" : VBF_ON_SUBSAMPLE_CUSTOMWEIGHTS[i], "files" : VBF_SUBSAMPLE_FILES[i]} for i in range(0, len(VBF_SUBSAMPLES)) if mass_poles[i] <= 180]

VBF_OFF_sample_data = {"basepath" : CUSTOM_PATH,
                  "subpaths" : VBF_OFF_subpath_dict
                 }

VBF_ON_sample_data = {"basepath" : CUSTOM_PATH,
                  "subpaths" : VBF_ON_subpath_dict
                 }

output_data["UL_VBF_OFF"] = VBF_OFF_sample_data
output_data["UL_VBF_ON"] = VBF_ON_sample_data
######################

##### ggH INFO #######
mass_poles = [125,160,170,180,190,200,210,230,250,270,300,350,400,450,500,550,600,700,800,900,1000,1500,2000,2500,3000]
ggH_SUBSAMPLES = ["ggH_"+str(mass) for mass in mass_poles]
ggH_SUBSAMPLE_SEARCH = {}
for mass_index, mass in enumerate(mass_poles):
    ggH_SUBSAMPLE_SEARCH[ggH_SUBSAMPLES[mass_index]] = ["*_GluGluHToWWTo2L2Nu_M" + str(mass) + "__*"]
ggH_SUBSAMPLE_WEIGHTS = [["XSWeight", "p_Gen_GG_SIG_kappaTopBot_1_ghz1_1_MCFM", "p_Gen_CPStoBWPropRewgt", "HWWOffshell_combineWgt"] for mass in mass_poles]
ggH_OFF_SUBSAMPLE_CUSTOMWEIGHTS = [["OFFSHELL_HIGGS"] for mass in mass_poles]
ggH_ON_SUBSAMPLE_CUSTOMWEIGHTS = [["ONSHELL_HIGGS"] for mass in mass_poles]

ggH_SUBSAMPLE_FILES = [[] for mass in mass_poles]

for subsample_index, subsample in enumerate(ggH_SUBSAMPLES):
    for search_q in ggH_SUBSAMPLE_SEARCH[subsample]:
        for yr_index, yr in enumerate(YEARS):
            file_path = CUSTOM_PATH + yr + "/" + STEPS_CUSTOM + STEPS[yr_index] + "/" + search_q + ".root"
            print(file_path)
            files = glob.glob(file_path)
            files = [f_name.split(CUSTOM_PATH)[-1] for f_name in files]
            ggH_SUBSAMPLE_FILES[subsample_index] = ggH_SUBSAMPLE_FILES[subsample_index] + files
ggH_OFF_subpath_dict = [{"name" : ggH_SUBSAMPLES[i], "weights" : ggH_SUBSAMPLE_WEIGHTS[i], "custom_weights" : ggH_OFF_SUBSAMPLE_CUSTOMWEIGHTS[i], "files" : ggH_SUBSAMPLE_FILES[i]} for i in range(0, len(VBF_SUBSAMPLES))]
ggH_ON_subpath_dict = [{"name" : ggH_SUBSAMPLES[i], "weights" : ggH_SUBSAMPLE_WEIGHTS[i], "custom_weights" : ggH_ON_SUBSAMPLE_CUSTOMWEIGHTS[i], "files" : ggH_SUBSAMPLE_FILES[i]} for i in range(0, len(VBF_SUBSAMPLES)) if mass_poles[i] <= 180]

ggH_OFF_sample_data = {"basepath" : CUSTOM_PATH,
                  "subpaths" : ggH_OFF_subpath_dict
                 }

ggH_ON_sample_data = {"basepath" : CUSTOM_PATH,
                  "subpaths" : ggH_ON_subpath_dict
                 }

output_data["UL_ggH_OFF"] = ggH_OFF_sample_data
output_data["UL_ggH_ON"] = ggH_ON_sample_data
######################



output_json = json.dumps(output_data, indent=4)
with open( output_json_name, "w") as outfile:
    outfile.write(output_json)





