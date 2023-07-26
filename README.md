## Environment

Create and activate the conda environment named `eeg2text` from the `environment.yaml` file.
```
conda env create -f environment.yaml
conda activate eeg2text
```

## Data

The datasets are not included in this repository. Please follow the instructions below to download and preprocess the datasets. Following [Wang and Ji, 2022](https://arxiv.org/abs/2112.02690), only sentiment and normal reading tasks are used.

The downloading instructions and preprocessing code have been adapted from [Open Vocabulary Electroencephalography-To-Text Decoding and Zero-shot Sentiment Classification](https://github.com/MikeWangWZHL/EEG-To-Text).

### Download the ZuCo Datasets

#### [ZuCo 1.0](https://osf.io/q3zws/)
Download the files for the following tasks from [OSF Storage v1.0](https://osf.io/q3zws/files/osfstorage):
- `task1-SR`
- `task2-NR`

> **_NOTE:_** The files are 43.6 GB, it can take some time to download.

Create the following directories in the repository's root directory:
- `dataset/ZuCo/task1-SR/Matlab_files`
- `dataset/ZuCo/task2-NR/Matlab_files`

Unzip the downloaded files and move the `.mat` files to their respective directories.

#### [ZuCo 2.0](https://osf.io/2urht/)
Download the file `task1-NR` from [OSF Storage v2.0](https://osf.io/2urht/files/).
> **_NOTE:_** The file is 62.2 GB.

Create the directory `dataset/ZuCo/task2-NR-2.0/Matlab_files` in the repository's root directory, unzip the downloaded file and move the `.mat` files to the created directory.

### Preprocess the Datasets

To preprocess the `.mat` files run the following command:
```
bash src/prepare_dataset.sh
```
> **_NOTE:_** Please be patient, it can take a bit of time to preprocess the files.

For each task, all `.mat` files will be converted into a single `.pickle` file and stored in the following path: `dataset/ZuCo/<task_name>/pickle/<task_name>-dataset.pickle`.
