# eric_bot email response in corporations | Doc2Vec | Multinomial Naiive Bayes 

A work in progress- estimated completion date June 1, 2023

Windows 10 PyCharm Community Edition 2022.2.3 

Python 3.10.10 

crontab 1.0.1

Genism 4.3.1

MatPlotLib 3.7.1

ntlk 3.8.1

numpy 1.24.2

pandas 2.0.0

Pillow 9.5.0

pip 23.1.2 

scikit_learn 1.2.2

scipy 1.10.1

seaborn 0.12.2

dataset for testing: https://www.kaggle.com/datasets/shashwatwork/consume-complaints-dataset-fo-nlp


User Guide
1.	Install necessary elements on a secure desktop with proper administrative access: 
o	PyCharm Community Edition 2022.2.3
o	Win10 Pro
o	Python 3.10.10
o	crontab 1.0.1
o	Genism 4.3.1
o	MatPlotLib 3.7.1 
o	ntlk 3.8.1 
o	numpy 1.24.2
o	pandas 2.0.0
o	Pillow 9.5.0
o	pip 23.1.2 
o	scikit_learn 1.2.2 
o	scipy 1.10.1
o	seaborn 0.12.2
o	SQLite3 3.7.15
o	Google chrome

1.	Operating System and approximate system requirements
Secure Personal Computer (Intel(R) Core(TM) i7-2600K CPU @ 3.40GHz   3.40 GHz, 8 GB RAM). 
2.	Open a chrome browser, log into mail.google.com using these credentials and keep the browser window open for later use. 
Email: XXXXX
Password: XXXXX
3.	Open PyCharm Community Edition 2022.2.3 > Git > clone> paste https://github.com/nicole1020/eric_bot/ and select clone. 
4.	Run the program with the green play button in the top right corner of the interface.
5.	Bar chart, Decision tree classifier, and confusion matrices will open independently. Please close each one so the program will continue running. *the decision tree and confusion matrices take a few minutes to load. Please wait for about 5 minutes while the program compiles.
6.	After about 6 minutes total the user interface will appear. You may select digits one through 7 for information retrieval or 8 to exit the program. 
7.	Upon selection the screen may pop up a login window to the gmail api confirmation page. Please continue back to Pycharm. *Known errors: if you receive the error “An error occurred: Authorized user info was not in the expected format, missing fields refresh_token.” Delete the file below and rerun the program.  Note it will only send one email per “run” on pycharm. 
eric_bot\Lib\site-packages\gensim\test\test_data\token.json 



