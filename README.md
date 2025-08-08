# 4Geeks data science project boilerplate

Minimal Python 3.11 repository for 4Geeks data science assignments. Several useful Python packages and VSCode extensions are installed on Codespace boot-up. Directories for models and data are created within the Codespace but excluded from git tracking.

## 1. Set-up

Fork this repository by clicking the *Fork* button at the upper right. Make sure to set 4Geeks as the owner of the new fork - this way 4Geeks pays for your codespace usage. Then start a Codespace on your fork by clicking the green *Code* button and then *Create codespace on main* under Codespaces in the drop-down menu.

## 2. Environment

### 2.1. Repository structure

```text
.
├──.devcontainer
│   └── devcontainer.json
│
├── src
│   └── project.ipynb
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

### 2.2. Python
**Base image**: [Python 3.11](https://github.com/devcontainers/images/tree/main/src/python)

Packages installed via `requirements.txt`:

1. [numpy 2.2.3](https://numpy.org/doc/stable/index.html)
2. [pandas 2.2.3](https://pandas.pydata.org/docs/)
3. [scikit-learn 1.6.1](https://scikit-learn.org/stable/index.html)
4. [matplotlib 3.10.1](https://matplotlib.org/stable/index.html)
5. [seaborn 0.13.2](https://seaborn.pydata.org/)
6. [ipykernel 6.29.5](https://pypi.org/project/ipykernel/)

If you need to install additional Python packages, you can do so via the terminal with: `pip install packagename`.

### 2.3. VSCode extensions

Sepcified via `devcontainier.json`.

1. [ms-python.python](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
2. [ms-toolsai.jupyter](https://marketplace.visualstudio.com/items?itemName=ms-toolsai.jupyter)
3. [streetsidesoftware.code-spell-checker](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker)

VSCode extensions can be added via the *Extensions* tab located on the activities panel at the left once inside the Codespace.
