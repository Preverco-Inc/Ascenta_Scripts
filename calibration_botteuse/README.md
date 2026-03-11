# Calibreation procedure for Botteuse

Script to calibrate Entre-Presse from multiple images


1. If not already done, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)
   - Once miniconda is installed follow instruction on 
   [this](https://medium.com/@arhamrumi/add-anaconda-to-path-on-windows-238dd77f1742) website to add conda to your PATH variables.
   - Restart your terminal after adding conda to PATH
   - Verify installation by running `conda --version` in your terminal
2. Download the repository
```bash
git clone https://github.com/Preverco-Inc/Ascenta_Scripts
```
3. Navigate to this folder (...\Ascenta_Script\Script_Ascenta) in cmd or powershell to create and activate the conda environment.

```bash
conda env update --prefix ./.condaenv --file environment.yml
conda activate ./.condaenv
```
5. Copy your calibration target images into a folder on your computer. 
6. Launch the script with the following command:

```bash
python main.py --input_folder "path\to\calib\target\image\folder" --output_calib_folder "path\to\output\calib\folder"
```

You can use this cmd to see all the options available for the script:

```bash
python main.py --help
```