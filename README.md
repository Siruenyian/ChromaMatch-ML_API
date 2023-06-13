# ChromaMatch-ML_API
Repository for ML api capstone project of Bangkit 2023

# Short Summary of API endpoint and how to use


## */predict* endpoint

### Input

- Image file with the name 'gambar'
- Color with the name 'rgb1'
- Color with the name 'rgb2'
- Color with the name 'rgb3'
- Color with the name 'rgb4'

Colors are in hexadecimal

example:

```
#B68A65
```

### Output

JSON response

example:
```avascript
{"season_from_color":"winter","tone_from_color":"olive","tone_from_image":"Olive"}
```

# Installing dependencies

Install venv

```bash
pip install virtualenv
```

Activate your virtual environment

```bash
./.venv/Scripts/activate.bat
```

Install from file

```bash
pip install -r requirements.txt
```

# Exporting dependencies

```bash
pip freeze > requirements.txt
```
