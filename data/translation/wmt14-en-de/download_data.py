import os
import subprocess
import platform


def check_install_wget():
    system = platform.system()
    try:
        # Try running wget with the --version option to check if it's available
        subprocess.run(['wget', '--version'], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("wget is not installed.")
        if system == "Darwin":  # macOS
            print("installing wget using Homebrew...")
            os.system('brew install wget')
        elif system == "Linux":
            print("installing wget using apt...")
            os.system('sudo apt install wget')
        else:
            print("Unsupported operating system.")
            exit(1)

# Check and install wget if necessary
check_install_wget()
current_dir = os.getcwd()
data_raw_dir = current_dir + '/raw'
# check if the data/raw folder exists
if os.path.exists(data_raw_dir):
    os.system('rm -r ' + data_raw_dir)
os.makedirs(data_raw_dir)
os.makedirs(data_raw_dir + '/train')
os.makedirs(data_raw_dir + '/test')
os.makedirs(data_raw_dir + '/vocab')
os.makedirs(data_raw_dir + '/dict')
os.makedirs(data_raw_dir + '/vocab_char')
# download the data into the data/raw folder
# train data
os.system('wget -P ' + data_raw_dir + '/train https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.en')
os.system('wget -P ' + data_raw_dir + '/train https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/train.de')

# test data
os.system('wget -P ' + data_raw_dir + '/test https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.en')
os.system('wget -P ' + data_raw_dir + '/test https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2012.de')
os.system('wget -P ' + data_raw_dir + '/test https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.en')
os.system('wget -P ' + data_raw_dir + '/test https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2013.de')
os.system('wget -P ' + data_raw_dir + '/test https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.en')
os.system('wget -P ' + data_raw_dir + '/test https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2014.de')
os.system('wget -P ' + data_raw_dir + '/test https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.en')
os.system('wget -P ' + data_raw_dir + '/test https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/newstest2015.de')

# vocab data
os.system('wget -P ' + data_raw_dir + '/vocab https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.en')
os.system('wget -P ' + data_raw_dir + '/vocab https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/vocab.50K.de')

# dict data
os.system('wget -P ' + data_raw_dir + '/dict https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de')

# vocab_char data
os.system('wget -P ' + data_raw_dir + '/vocab_char https://nlp.stanford.edu/projects/nmt/data/wmt14.en-de/dict.en-de')
