
pip install waitress

python -m venv venv
source venv/bin/activate  # On Linux/Mac
venv\Scripts\activate     # On Windows
pip install -r requirements.txt


Run the Flask Server:
python <your_script_name>.py


venv\Scripts\activate
deactivate



build Docker Image:

docker build -t remove-bg .

run container:
docker run -p 5000:5000 remove-bg
