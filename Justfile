VENV := justfile_directory() + "/.venv"
PYTHON := justfile_directory() + "/.venv/bin/python"
PKG := "pure-cv"

help:
  @just --list

# Build the package (into `dist/`)
build:
  cd {{PKG}} && \
  rm -drf dist && \
  {{PYTHON}} -m build && \
  rm -drf build

# Publish `dist/*` to pypi, then delete
publish:
  cd {{PKG}} && \
  {{PYTHON}} -m twine upload dist/* && \
  rm -drf dist

# Increase patch version
patch:
  $CIT_SCRIPTS/bump.sh {{PKG}}/pyproject.toml

# Build and publish
republish: patch build publish

init:
  rm -drf {{VENV}} || :
  python3.11 -m venv {{VENV}}
  {{PYTHON}} -m pip install --upgrade pip
  {{PYTHON}} -m pip install -r requirements.txt