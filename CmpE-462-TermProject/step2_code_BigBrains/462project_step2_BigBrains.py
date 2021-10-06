from naive_bayes import naive_bayes
import sys
import utils

TRAIN_PATH = utils.get_abs_path('./TRAIN')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Invalid command. Try: python3 462project_step2_TeamA.py step2_model_TeamA.pkl <test-dataset-folder>")
    else:
        _, pickle_path, test_dataset_folder = sys.argv
        naive_bayes(TRAIN_PATH, test_dataset_folder, pickle_path)
