# gg-project-master
Golden Globe Project Master

# Methodology overview

For an overview of our methodologies please view the overview.pptx or overview.pdf file.

# Packages
The following packages are required to run the code

### NLTK
sudo pip install -U nltk  
https://www.nltk.org/install.html

### Pandas
pip install pandas   
https://pandas.pydata.org/

### Numpy
sudo pip install -U numpy  
https://www.numpy.org/

### Unidecode
pip install Unidecide  
https://pypi.org/project/Unidecode/

### Requests
pip install requests  
http://docs.python-requests.org/en/master/

# Run Instructions

The following instructions apply to running analysis on years 2013, 2015, 2018, and 2019.


Pre-ceremony call - this loads in the knowledge base for our gg_api.

`python3 -c 'import gg_api; gg_api.pre_ceremony()'`

Runs the gg_api which generates our results.

`python3 -c 'import gg_api; gg_api.main()'`

Runs the autograder on our results.

`python3 autograder.py`


## Additional Notes

english.txt is a text file of common stopwords from https://github.com/Alir3z4/stop-words that is more comprehensive than the NLTK list of stopwords.
