
### INSTALL REQUIREMENTS ###
#Create a virtualenv#
python3.6 -m venv .venv

#activate venv#
source .venv/bin/activate

#To setup the tool, install the dependencies:#
pip install -r requirements.txt

### RUN CODE ###
#to run the code, launch the following command:#
python main.py /path/to/file/file_name.root path/to/output/
example command: python main.py 50K_Merged_Vtx_sample.root out_test/
