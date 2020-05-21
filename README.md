# se20-project-16

To install requirements on Linux run (in project root folder):
>pip3 install -r src/requirements.txt

Per usare nltoolkit serve lanciare una command line python (python3 in terminale Linux) e runnare:
>import nltk  
>
>nltk.download('punkt')  
>nltk.download('averaged_perceptron_tagger')  
>nltk.download('maxent_ne_chunker')  
>nltk.download('words')
>
Si può considerare di importare tutti i tools di nltk con
>nltk.download('all')