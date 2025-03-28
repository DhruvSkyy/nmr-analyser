## Installation

The program requires **Python 3.12.8** and the following external libraries: `NumPy == 2.2.2`, `Matplotlib == 3.10.0`, and `scikit-learn == 1.6.1`. These can be installed using the following command:

```bash
pip install numpy==2.2.2 matplotlib==3.10.0 scikit-learn==1.6.1
```

Once installed, the program can be run from the command line. Ensure that the script is placed in the directory from which you wish to execute it.



## Usage

The program is a command-line tool designed for **NMR peak analysis**. It processes a CSV file containing spectral data and determines peak multiplicities based on clustering and J-value analysis.

To run the program, use the following command:

```bash
python main.py --file [input_file] --frequency [MHz] [options]
```

Replace `[input_file]` with the path to the CSV file containing the NMR data and `[MHz]` with the operating frequency of the spectrometer.



## Command-line Arguments

The program accepts the following arguments:

- `--file`: **(Required)** Path to the input CSV file containing the NMR spectral data. The file should include columns for chemical shift (ppm) and intensity.
- `--frequency`: **(Required)** Operating frequency of the spectrometer in MHz. Used to convert peak separations from ppm to Hz for J-value calculations.
- `--extra`: *(Optional, default: 0)*  
  Specifies how many additional clusters the program should test beyond the automatically detected number.  
  This is useful when closely and widely spaced clusters are both present. The program will iteratively test additional clusters and select the configuration with the fewest misclassified peaks.
- `--uncertainty`: *(Optional, default: 1.0)*  
  Defines the tolerance for matching calculated J-values to expected values.  
  A larger value increases flexibility in multiplicity classification, while a smaller value enforces stricter matching.



## Example Usage

For an NMR dataset stored in `tests/hmdb_sample_4_f600.tsv`, recorded at 600 MHz:

```bash
python "src/main.py" --file "tests/hmdb_sample_4_f600.tsv" --frequency 600
```

To allow the program to test 10 additional clusters and reduce the uncertainty tolerance to 0.1:

```bash
python "src/main.py" --file "tests/hmdb_sample_4_f600.tsv" --frequency 600 --uncertainty 0.1 --extra 10
```

Note: The test data in the tests/ directory follows the naming convention fXXX, where XXX denotes the recording frequency in MHz (e.g. f600 = 600 MHz). The initial execution of the programme may require a few seconds to complete.

## Output

The program produces four outputs:

1. `simple_calculations.txt`: Summarised detected peaks with their predicted multiplicities, J values and possible functional groups.
2. `detailed_calculations.txt`: Detailed output including all peak positions, all J-values, and clustering information.
3. A graphical output showing the spectrum with detected peaks and multiplicities.
4. Terminal output for quick visual inspection.

All output files are saved in the nmrdata_output/ directory. If the directory or files already exist, they are automatically overwritten. The outputs are named based on the original input filename.

## Example Output

**Test 4 Terminal Output:**

```text
¹H NMR (600 MHz)
δ 2.85 ppm (t, J = 7.25 Hz)
δ 3.21 ppm (t, J = 7.25 Hz)
δ 6.74 ppm (dd, J = 2.09, 8.13 Hz)
δ 6.84 ppm (d, J = 2.09 Hz)
δ 6.89 ppm (d, J = 8.13 Hz)
```

**Generated Files:**

- `example_output/hmdb_sample_4_f600_nmr_plot.jpg`
- `example_output/hmdb_sample_4_f600_simple_calculation.txt`
- `example_output/hmdb_sample_4_f600_detailed_calculation.txt`

Note: All generated example files are stored in the example_output folder. 