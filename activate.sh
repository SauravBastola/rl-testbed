# You must run
#   source ./activate
# from shell

if ! [ -f .venv/bin/activate ]; then
    echo "You haven't created a virtual environment"

    if ! command -v pyenv &> /dev/null ; then
        echo "Also, pyenv is not installed."
        echo "For MacOS \n\t brew install pyenv"
        echo "For Linux check your system's package manager. \n\t apt install pyenv"
        echo "After that setup your shell environment variables \n\t pyenv init"
        echo "Now install python3.11 \n\t pyenv install 3.11"
    fi

    echo "Use python3.11 to create a virtual environment \n\t pyenv shell 3.11"
    echo "\t python -m venv .venv"
    return
fi

source .venv/bin/activate

python --version | grep 3.11 &> /dev/null
if [ $? -eq 1 ]; then
    echo "Python version 3.11 is required."
    deactivate
fi
