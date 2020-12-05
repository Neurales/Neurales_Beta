  import os

def install_function():
    os.system('pip3 uninstall setuptools')
    os.system('pip3 uninstall Cython')
    os.system('pip3 uninstall wheel')
    os.system('pip3 uninstall torch torchvision torchaudio')
    os.system('pip3 uninstall appnope backcall cycler decorator ipython ipython-genutils jedi joblib kiwisolver matplotlib nltk numpy opencv-python pandas parso pexpect pickleshare pillow prompt-toolkit ptyprocess Pygments pyparsing python-dateutil pytz==2019.3 scikit-learn scipy six sklearn tqdm traitlets wcwidth')
   


install_function()
