
# Home Security Surveillance

This project uses human detection together with facial recognition.


## Run Locally

Clone the project

```bash
  git clone https://github.com/JDSNX/home-security-surveillance.git
```

Go to the project directory

```bash
  cd home-security-surveillance
```

Install virtual environment and create virtual environment

```bash
  pip install --upgrade virtualenv
  py -m venv /path/to/new/virtual/environment
```

Start virtual environment

```bash
  ./venv/Scripts/activate
```

Install dependencies

```bash
  pip install -r requirements.txt
```

### Run and Train

To train collected datasets

```bash
  cd src/
  py main.py train
```

To run both facial recognition and human detection

```bash
  cd src/
  py main.py run
```



## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`MONGODB` - MONGODB SERVER

`DB` - COLLECTION

`YOLO_WEIGHTS` - PATH FOR YOLO_WEIGHTS

`YOLO_CFG` - PATH FOR YOLO_CFG

`CLASSES` - PATH FOR LIST OF CLASSES

`ENCODINGS` - PATH FOR .PICKLE FILE
## Acknowledgements

 - [Face Recognition by ageitgey](https://github.com/ageitgey/face_recognition)
