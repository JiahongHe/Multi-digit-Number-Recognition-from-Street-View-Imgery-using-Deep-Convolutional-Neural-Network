echo Downloading train dataset
wget http://ufldl.stanford.edu/housenumbers/train.tar.gz
echo Downloading test dataset
wget http://ufldl.stanford.edu/housenumbers/test.tar.gz
echo Downloading extra dataset
wget http://ufldl.stanford.edu/housenumbers/extra.tar.gz

echo Extracting dateset files
tar -xvzf train.tar.gz
tar -xvzf test.tar.gz
tar -xvzf extra.tar.gz

