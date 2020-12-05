import os

def install_function():
    os.system('wget https://nvidia.box.com/shared/static/yr6sjswn25z7oankw8zy1roow9cy5ur1.whl -O torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl')
    os.system('sudo apt-get install python3-pip libopenblas-base libopenmpi-dev')
    os.system('sudo pip3 install Cython')
    os.system('sudo pip3 install torch-1.6.0rc2-cp36-cp36m-linux_aarch64.whl')
    os.system('sudo apt-get install libjpeg-dev zlib1g-dev')
    os.system('git clone --branch v0.6.0 https://github.com/pytorch/vision torchvision')
    os.system('cd ~/torchvision')  
    os.system('sudo python3 setup.py install')
    os.system('cd ~/')
    os.system('sudo apt-get install -y python3-pip libopenblas-base libopenmpi-dev build-essential cmake git libgtk2.0-dev libgtk-3-dev pkg-config libgstreamer-plugins-base1.0-dev libgstreamer1.0-dev libavcodec-dev libavformat-dev libswscale-dev python3-dev python3-numpy python3-pandas libtbb2 libtbb-dev libjpeg-dev libpng-dev libtiff-dev libdc1394-22-dev gcc g++ python-opencv')
    os.system('sudo pip3 install setuptools==50.3.0 ')
    os.system('sudo apt-get install -y libpcap-dev libpq-dev ')
    os.system('sudo apt-get install -y libjpeg-dev zlib1g-dev ')
    os.system('sudo pip3 install appnope==0.1.0  apturl==0.5.2 asn1crypto==0.24.0 backcall==0.2.0 beautifulsoup4==4.6.0 blinker==1.4 Brlapi==0.6.6 certifi==2020.6.20 chardet==3.0.4 click==7.1.2 cryptography==2.1.4 cupshelpers==1.0 cycler==0.10.0 decorator==4.4.2 defer==1.0.6 distro-info==0.18ubuntu0.18.04.1 feedparser==5.2.1 future==0.18.2 html5lib==0.999999999 httplib2==0.9.2 idna==2.6 ipython==7.16.1 ipython-genutils==0.2.0 jedi==0.17.2 joblib==0.16.0 keyring==10.6.0 keyrings.alt==3.0 kiwisolver==1.2.0 language-selector==0.1 launchpadlib==1.10.6 lazr.restfulclient==0.13.5 lazr.uri==1.0.3 louis==3.5.0 lxml==4.2.1 macaroonbakery==1.1.3 Mako==1.0.7 MarkupSafe==1.0 matplotlib==3.3.2 nltk==3.5 numpy==1.17.2 oauth==1.0.1 oauthlib==2.0.6 olefile==0.45.1 pandas==0.25.1 parso==0.8.0 pexpect==4.8.0 pickleshare==0.7.5 Pillow==7.2.0 pip==9.0.1 prompt-toolkit==3.0.7 protobuf==3.0.0 ptyprocess==0.6.0 pycairo==1.16.2 pycrypto==2.6.1 pycups==1.9.73 Pygments==2.7.1 pygobject==3.26.1 PyICU==1.9.8 PyJWT==1.5.3 pymacaroons==0.13.0 PyNaCl==1.1.2 pyparsing==2.4.7 pyRFC3339==1.0 python-apt==1.6.5+ubuntu0.3 python-dateutil==2.8.1 python-debian==0.1.32 pytz==2020.1 pyxdg==0.25 PyYAML==3.12 regex==2020.7.14 requests==2.18.4 requests-unixsocket==0.1.5 scikit-learn==0.23.2 scipy==0.19.1 SecretStorage==2.3.1 simplejson==3.13.2 six==1.15.0 ssh-import-id==5.7 system-service==0.3 systemd-python==234 threadpoolctl==2.1.0 tqdm==4.49.0 traitlets==4.3.3 ubuntu-drivers-common==0.0.0 uff==0.6.9 urllib3==1.22 urwid==2.0.1 wadllib==1.3.2 wcwidth==0.2.5 webencodings==0.5 wheel==0.30.0 xkit==0.0.0 zope.interface==4.3.2')
    os.system('sudo pip3 install --upgrade numpy==1.17.2 pandas==0.25.1')
    os.system('sudo apt-get install -y jupyter')
    os.system('sudo apt-get install python3-sklearn python3-sklearn-lib')
    
install_function()
