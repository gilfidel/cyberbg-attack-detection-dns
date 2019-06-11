# cyberbg-attack-detection-dns
DNS based user profiling project

## Usage
0. Using python37
1. in src folder: pip install -r requirements.txt
2. preprocess dataset by running: _python main.py preprocess /path/to/data/dir /path/to/preprocessed_data/dir_

   data dir should contain all the original csv files
   
   preprocessing basically runs Asa's preprocessor and saves the results to a new csv
3. run training with grid param search: _python main.py run /path/to/preprocessed_data/dir_
