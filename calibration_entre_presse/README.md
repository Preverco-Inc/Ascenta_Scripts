# Script

Script to calibrate Entre-Presse from multiple images


2. If not already done, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)
   - Once miniconda is installed follow instruction on 
   [this](https://medium.com/@arhamrumi/add-anaconda-to-path-on-windows-238dd77f1742) website to add conda to your PATH variables.
   - Restart your terminal after adding conda to PATH
   - Verify installation by running `conda --version` in your terminal
1. Download the repository
```bash
git clone 
```
4. Navigate to this folder in cmd or powershell to create and activate the conda environment.

```bash
conda env update --prefix ./.condaenv --file environment.yml
conda activate ./.condaenv
```

4. Launch the script with the following command:

```bash
python calibration_entre_presse.py --input_folder "path\to\calibration\target\image\folder"