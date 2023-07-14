## Environment setup

Create and activate conda environment named ```eeg2text``` from the ```environment.yaml``` file.
```sh
conda env create -f environment.yaml
conda activate eeg2text
```

## Download and preprocess ZuCo datasets

Adapted from [Open Vocabulary Electroencephalography-To-Text Decoding and Zero-shot Sentiment Classification](https://github.com/MikeWangWZHL/EEG-To-Text).

### Download datasets

#### [ZuCo 1.0](https://osf.io/q3zws/)
Download the files for the following tasks from the [OSF Storage](https://osf.io/q3zws/files/osfstorage):
-`task1-SR`
-`task2-SR`
-`task3-TSR`

*Note: The files are 63.7 GB and it takes about an hour to download.

Create the following directories in the repository's root directory:
-`/dataset/ZuCo/task1-SR/Matlab_files`
-`/dataset/ZuCo/task2-NR/Matlab_files`
-`/dataset/ZuCo/task3-TSR/Matlab_files`

Unzip the downloaded files and move the `.mat` files to their respective directiories

#### [ZuCo 2.0](https://osf.io/q3zws/](https://osf.io/2urht/)
- Download the file `task1-NR` from the [OSF Storage](https://osf.io/2urht/files/)
- Create the directory `/dataset/ZuCo/task2-NR-2.0/`
- Unzip the download and move the `.mat` files to their directiory above

### Preprocess datasets
modified instructions here
