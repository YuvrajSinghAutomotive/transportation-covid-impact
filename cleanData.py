'''
Download Dataset from Kaggle
'''
import kaggle
import zipfile
import os
os.system('kaggle datasets download -d sobhanmoosavi/us-accidents')
zipfile.ZipFile('us-accidents.zip').extractall()

