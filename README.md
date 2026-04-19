# Grill Assignment — Setup

This repository requires native build tools for `dlib` (used by `face_recognition`). Follow the steps below for Windows.

1) Recommended: use the existing virtual environment (or create one):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) Install native build tools (Windows):
- Install CMake (add to PATH). Use winget:

```powershell
winget install --id Kitware.CMake
```

- Install Visual Studio Build Tools (C++ workload):

```powershell
winget install --id Microsoft.VisualStudio.2022.BuildTools
```

3) Install Python dependencies into the activated venv:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If building `dlib` still fails, consider using Conda which provides prebuilt binaries:

```bash
conda create -n fr python=3.10
conda activate fr
conda install -c conda-forge dlib face_recognition opencv imutils
```
Notes:
- On Windows, adding CMake to PATH during install is required so pip can find it.
- Building `dlib` can take time; the Conda route avoids building from source.