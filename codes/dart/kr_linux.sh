sudo apt-get install language-pack-ko
sudo locale-gen ko_KR.UTF-8
sudo dpkg-reconfigure locales
sudo update-locale LANG=ko_KR.UTF-8 LC_MESSAGES=POSIX

# unfonts
sudo apt-get install fonts-unfonts-core fonts-unfonts-extra 

# baekmuk
sudo apt-get install fonts-baekmuk

#nanum
sudo apt-get install fonts-nanum fonts-nanum-coding fonts-nanum-extra