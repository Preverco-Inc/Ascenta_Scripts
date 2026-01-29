# Calibreation procedure for Entre-Presse

Script to calibrate Entre-Presse from multiple images

1. If not aleready installed, install AIL (Aurora Imaging Library) or AIL Lite on your computer.
   - Go to the [download page](https://www.zebra.com/us/en/support-downloads/software/industrial-machine-vision-and-fixed-scanners-software/aurora/aurora-imaging-library.html)
   on Zebra website.
   - Download Aurora Imaging Library Lite X Version 24H2 (10.7)
   - Once downloaded, run the installer and follow the instructions.
2. If not already done, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#windows-installation)
   - Once miniconda is installed follow instruction on 
   [this](https://medium.com/@arhamrumi/add-anaconda-to-path-on-windows-238dd77f1742) website to add conda to your PATH variables.
   - Restart your terminal after adding conda to PATH
   - Verify installation by running `conda --version` in your terminal
3. Download the repository
```bash
git clone https://github.com/Preverco-Inc/Ascenta_Scripts
```
4. Navigate to this folder (...\Ascenta_Script\Script_Ascenta) in cmd or powershell to create and activate the conda environment.

### MIL API NOT WORKING Waiting for return from Zebra support

```bash
conda env update --prefix ./.condaenv --file environment.yml
conda activate ./.condaenv
```
5. Copy your calibration target images into a folder on your computer. 
6. Launch the script with the following command:

```bash
python main.py --input_folder "path\to\calib\target\image\folder" --output_calib_folder "path\to\output\calib\folder"
```