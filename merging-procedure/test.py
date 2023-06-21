import json

hww_wgt_file = open("./ggH_output_preprocessing.json")
hww_wgt_contents = hww_wgt_file.read()
hww_wgts = json.loads(hww_wgt_contents)

sample_json_id = "HM_200"

renormWgt = hww_wgts[sample_json_id]["RENORM_WGT"]
combineWgt = hww_wgts[sample_json_id]["COMB_WGTS"]

lossCompWgt = hww_wgts[sample_json_id]["LOSS_COMP"]

cutoffSIG = hww_wgts[sample_json_id]["SIG_CUTOFF"]
cutoffCONT = hww_wgts[sample_json_id]["CONT_CUTOFF"]
cutoffSIGplusCONT = hww_wgts[sample_json_id]["SIGplusCONT_CUTOFF"]
print("RENORM WGT: " + str(renormWgt))
print("COMB WGT:" + str(combineWgt))
print("LOSS COMP:" + str(lossCompWgt))
