## Environment

Create and activate the conda environment named `eeg2text` from the `environment.yaml` file.
```sh
conda env create -f environment.yaml
conda activate eeg2text
```

## Data

The datasets are not included in this repository. Please follow the instructions below to download and preprocess the datasets. Adapted from [Open Vocabulary Electroencephalography-To-Text Decoding and Zero-shot Sentiment Classification](https://github.com/MikeWangWZHL/EEG-To-Text).

### Download the ZuCo Datasets

#### [ZuCo 1.0](https://osf.io/q3zws/)
Download the files for the following tasks from [OSF Storage v1.0](https://osf.io/q3zws/files/osfstorage):
- `task1-SR`
- `task2-SR`
- `task3-TSR`

* Note: The files are 63.7 GB and it takes about an hour to download.

Create the following directories in the repository's root directory:
- `/dataset/ZuCo/task1-SR/Matlab_files`
- `/dataset/ZuCo/task2-NR/Matlab_files`
- `/dataset/ZuCo/task3-TSR/Matlab_files`

Unzip the downloaded files and move the `.mat` files to their respective directories.

#### [ZuCo 2.0](https://osf.io/2urht/)
- Download the file `task1-NR` from [OSF Storage v2.0](https://osf.io/2urht/files/).
- Create the directory `/dataset/ZuCo/task2-NR-2.0/`.
- Unzip the downloaded file and move the `.mat` files to the directory above.

### Preprocess the Datasets
modified instructions here
