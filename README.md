<h2><u> GIRAFE </u></h2>

**G**lottal **I**maging **R**epository for **A**dvanced Segmentation, Analysis, and **F**ast **E**valuation

<img src="GIRAFE.png" alt="GIRAFE" width="300"/>

Welcome to the GIRAFE Database repository! This is a comprehensive collection of code and data used for cleaning and calculating the CUCO database. The database comprises recordings of patients and control individuals, with voice samples taken at three different time points: 2 weeks before surgery, 2 weeks after surgery, and 3 months after surgery. The surgeries include FESS, Tonsillectomy, Septoplasty, and a control group.

<h2><u>Repository Structure</u></h2>

1. **DL_code folder:**

  
2. **Matlab_code folder:**


3. **Matlab_code.ipynb:**


4. **Seg_FP-Results.ipynb:**


<h2><u>Quick Start</u></h2>

The **GIRAFE** Database is available on Zenodo with a Digital Object Identifier (DOI) to ensure easy access and citation. To access the database, follow these steps:

Visit the Zenodo page for the **GIRAFE** Database using the following link: GIRAFE Database on Zenodo
You can download the dataset files directly from Zenodo.

For more information and specific setup instructions, refer to the dataset documentation on Zenodo.

## Installation

To set up the project using Conda, follow these steps:

```bash
# Clone the repository
https://github.com/Andrade-Miranda/GIRAFE.git
cd GIRAFE

# Create a Conda environment from the .yml file
conda env create -f environment.yml

# Activate the Conda environment
conda activate your-env-name
```

## Usage

Each script can be run independently, depending on the specific analysis you wish to perform. Here are the general steps to follow:

1. **Prepare Your Data:** Ensure the GIRAFE database is available either inside the GIRAFE repository or in any other location.
2. **Configure Hyperparameters:** Adjust models hyperparameters inside spripts.
3. **Run the Script:** Execute the script using a Python interpreter. For example:

   ```bash
   python DL_code/train.py 
   python DL_code/inference.py --model_dir Unet_8_100_0.0002_256_Baseline
   ```
`training.py` and `inference.py` scripts have the data_dir path set to the default value ../GIRAFE, but you can change it using the argument --data_dir <GIRAFE path>. After training, `training.py` generates a ./DL_code/Results directory where the models are saved. `inference.py` requires the innest directory name containing the model to be passed as argument using --model_dir.

<h2><u>How to Cite</u></h2>
If you use the CUCO Database in your research or projects, we kindly request that you cite it to give credit to the contributors. Please use the following references to cite the database:

1. **Zenodo Dataset**:  
   To cite the dataset available on Zenodo, use the provided DOI:
   > Hernández-García, E., Guerrero-López, A., Arias-Londoño, J. D., & Godino Llorente, J. I. (2024). CUCO Database: A voice and speech corpus of patients who underwent upper airway surgery in pre- and post-operative states [Data set]. Zenodo. [https://doi.org/10.5281/zenodo.10256802](https://doi.org/10.5281/zenodo.10256802)

2. **Scientific Data Paper**:  
   Additionally, cite the associated paper in the *Scientific Data* journal where the database is described in detail:
   > Hernández-García, E., Guerrero-López, A., Arias-Londoño, J.D. et al. A voice and speech corpus of patients who underwent upper airway surgery in pre- and post-operative states. *Sci Data* 11, 746 (2024). [https://doi.org/10.1038/s41597-024-03540-5](https://doi.org/10.1038/s41597-024-03540-5)

Citing both the Zenodo dataset and the journal paper helps acknowledge the work of the contributors and ensures proper recognition in the academic community.


<h2><u>Usage</u></h2>

<h2><u>License</u></h2>

<h2><u>Credits</u></h2>

<h2><u>Contact Information</u></h2>
