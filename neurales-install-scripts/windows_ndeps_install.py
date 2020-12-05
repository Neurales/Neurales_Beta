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
    os.system('pip3 install torch===1.7.0 torchvision===0.8.1 torchaudio===0.7.0 -f https://download.pytorch.org/whl/torch_stable.html')
    os.system('pip3 install appnope==0.1.0 backcall==0.1.0 cycler==0.10.0 decorator==4.4.0 ipython==7.9.0 ipython-genutils==0.2.0 jedi==0.15.1 joblib==0.14.0 kiwisolver==1.1.0 nltk==3.4.5 opencv-python==4.1.2.30 parso==0.5.1 pexpect==4.7.0 pickleshare==0.7.5 prompt-toolkit==2.0.10 ptyprocess==0.6.0 Pygments==2.4.2 pyparsing==2.4.2 python-dateutil==2.8.0 pytz==2019.3 scikit-learn==0.21.3 six==1.12.0 sklearn==0.0  tqdm==4.36.1 traitlets==4.3.3 wcwidth==0.1.7')
   


install_function()
