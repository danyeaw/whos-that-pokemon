# Who's That Pokémon

A Pokémon Trading Card scanner using OpenCV.

It currently only supports the US-English
[Scarlet & Violet—Stellar Crown](https://tcg.pokemon.com/en-us/galleries/stellar-crown/)
expansion set, but others could easily be added.

## Usage

The app is hosted on PyScript.com:
https://dyeaw.pyscriptapps.com/who-s-that-pokemon/

## Development

You can fork the PyScript app by creating an account and accessing it here: https://pyscript.com/@dyeaw/who-s-that-pokemon/

### Development Environment

A development environment is needed to interact with the notebook and create the JSON card database.

#### Python with pip:

##### Linux and macOS

```bash
$ python -m venv .venv
$ source .venv/bin/activate
$ pip install -r requirements.txt
```

##### Windows:

```PowerShell
> python -m venv .venv
> source .venv/Scripts/activate.ps1
> pip install -r requirements.txt
```

#### Conda

```
$ conda env create -f environment.yml 
$ conda activate pokemon
```

## Credit

OpenCV Card Scanning ideas and Magmar test image from:
https://github.com/NolanAmblard/Pokemon-Card-Scanner

Image Hashing algorithms partially based on:
https://github.com/JohannesBuchner/imagehash
