from bs4 import BeautifulSoup
import re

data = open('../data/kaz-rus/kaz-rus.data','r')

train_rus= open("train.tags.kaz-rus.rus","w+")
train_kaz= open("train.tags.kaz-rus.kaz","w+")

dev_rus = open("valid.tags.kaz-rus.rus","w+")
dev_kaz = open("valid.tags.kaz-rus.kaz","w+")


test_rus = open("test.tags.kaz-rus.rus","w+")
test_kaz = open("test.tags.kaz-rus.kaz","w+")

counter = 0
for line in data.readlines():
   # line =  re.split(r'\s{3,}', line)
   # print(line)
    if counter % 1680 == 0:
      test_kaz.write(line.split('\t')[0].replace('\n',' ').strip() + '\n')
      test_rus.write(line.split('\t')[1].replace('\n',' ').strip() + '\n')
    elif counter % 1681 == 0:
      dev_kaz.write(line.split('\t')[0].replace('\n',' ').strip() + '\n')
      dev_rus.write(line.split('\t')[1].replace('\n',' ').strip() + '\n')
    else:  
      train_kaz.write(line.split('\t')[0].replace('\n',' ').strip() + '\n')
      train_rus.write(line.split('\t')[1].replace('\n',' ').strip() + '\n')
    counter += 1

train_kaz.close()
train_rus.close()


test_kaz.close()
test_rus.close()


dev_kaz.close()
dev_rus.close()

from nltk.tokenize import WordPunctTokenizer
def tokenize(src_filename, new_filename):
  with open(src_filename, encoding="utf-8") as src_file:
    with open(new_filename, "w", encoding="utf-8") as new_file:
      for line in src_file:
        new_file.write("%s" % ' '.join(WordPunctTokenizer().tokenize(line)))
        new_file.write("\n")

        
#train
tokenize("train.tags.kaz-rus.rus", "train.tok.kaz-rus.rus")
tokenize("train.tags.kaz-rus.kaz", "train.tok.kaz-rus.kaz")


tokenize("test.tags.kaz-rus.rus", "test.tok.kaz-rus.rus")
tokenize("test.tags.kaz-rus.kaz", "test.tok.kaz-rus.kaz")

tokenize("valid.tags.kaz-rus.rus", "valid.tok.kaz-rus.rus")
tokenize("valid.tags.kaz-rus.kaz", "valid.tok.kaz-rus.kaz")
