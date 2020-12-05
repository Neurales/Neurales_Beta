import os

def install_function():
    os.system('pip3 install setuptools')
    os.system('pip3 install Cython')
    os.system('pip3 install wheel')
    os.system('pip3 install matplotlib') #matplotlib==3.1.1
    os.system('pip3 install numpy') #numpy==1.17.2
    os.system('pip3 install pandas') #pandas==0.25.1
    os.system('pip3 install pillow') #Pillow==6.2.0
    os.system('pip3 install scipy') #scipy==1.3.1
    os.system('pip3 install torch torchvision torchaudio -f https://download.pytorch.org/whl/torch_stable.html')
    os.system('pip3 install appnope backcall cycler decorator ipython ipython-genutils jedi joblib kiwisolver nltk opencv-python parso pexpect pickleshare prompt-toolkit ptyprocess Pygments pyparsing python-dateutil pytz scikit-learn six sklearn tqdm traitlets wcwidth')
   


install_function()
