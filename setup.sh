# Create a virtual environment
python3 -m venv simple-text-classification-venv

# Activate the virtual environment
source ./simple-text-classification-venv/bin/activate

# Install requirements
python3 -m pip install --upgrade pip
python3 -m pip install -r ./requirements.txt

# deactivate
deactivate

# To remove the virtual environment run the following command in the terminal
#rm -rf simple-text-classification-venv