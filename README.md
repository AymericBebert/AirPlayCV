# AirPlayCV

Project idea: use the computer webcam to be able to play in the air and produce sound

## Setup

Developed with Python 3.7 and should be compatible with Python 3.6+

Developed on macOS. Not sure about compatibility with other systems

Install the requirements with the usual:

```sh
pip3 install -r requirements.txt
```

This project uses some Cython code that needs to be built.
In your terminal, go to the project root folder and run:

```sh
./src/cyvisord/cython-setup.py build_ext --inplace
```

## Play

You should be able to play with:

```sh
./src/main.py
```

You may have to run `export PYTHONPATH=$(pwd)` beforehand for the imports to work.

When you are done playing, press `q` while the window in on focus to quit.
