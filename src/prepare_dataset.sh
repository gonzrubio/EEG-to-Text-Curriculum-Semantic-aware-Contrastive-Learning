echo "This script constructs .pickle files from .mat files from the ZuCo datasets."
echo "Note: This process can take time, so please be patient..."

python3 src/data/construct_dataset_mat_to_pickle_v1.py -t task1-SR
python3 src/data/construct_dataset_mat_to_pickle_v1.py -t task2-NR
python3 src/data/construct_dataset_mat_to_pickle_v2.py

