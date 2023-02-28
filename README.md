## HWWOffshellTools
Tools used in the HWW Offshell analysis. 

## Setup

### For Running
```
source /cvmfs/cms.cern.ch/cmsset_default.sh #Required on CMSLPC
cmsrel CMSSW_10_6_19_patch2
cd CMSSW_10_6_19_patch2/src

git clone git@github.com:jrotter2/HWWOffshellTools.git
cd HWWOffshellTools

pip install -r requirements.txt --user
```

### Configurations
To modify input and output files, the `config` directory holds a `dataset_config.txt` which stores settings for each dataset including the list of files for each sample and their EOS paths.

## Merging Procedure
The mergining procedure has been condensed into one script called `preprocessing_samples.py` which will compute the merging, loss compensation, and renormalization weights described in `AN2021-149`.

The script can be run using,
```
cd merging-procedure
python preprocessing_samples.py
```
Histograms after each step in the procedure will be stored in `output_preprocessing.root`, 2d plots will be stored in `output_preprocessing.pdf`, and all computed weights will be stored in `output_preprocessing.json` to be used in the Latinos Module. 

