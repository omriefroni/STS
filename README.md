# STS
Official implementation of our work as described in:
"STS - Spectral Teacher Spatial Student for dense shape correspondence - Spectrum-Aware Real-Time Dense Shape Correspondence" (3DV 2022)


## Installation

It is recommended to use a virtual environment for each student model.

For STS_CorrNet3D follow: [CorrNet3D/README.md](CorrNet3D/README.md) installation (has additional installation for STS).

For STS_DPC: 
Please follow `DPC/installation.sh` or simply run (has additional installation for STS).
```
bash DPC/installation.sh 
```

## Dataset 
Spectral datasets for training with spectral teacher, can be downloaded from [here](https://drive.google.com/drive/u/1/folders/1S5fp8QN_rBWUbwHmmyVFhpeQLR99QsN2). 
Put the folder under Spectral_data directory (Spectral_data\datasets\<dataset_name>. Datasets: \[SHREC, FAUST, SURREAL\])



## STS_CorrNet3D 

The code is based on [CorrNet3D official implementation](https://github.com/ZENGYIMING-EAMON/CorrNet3D), with changes for the additional spectral teacher. 

Default training contains the spectral teacher with spectral datasets.
For testing use CorrNet3D testset - [CorrNet3D/README.md](CorrNet3D/README.md).
Testing with geodesic distance metric will be updated soon... 
