from bs4 import BeautifulSoup
data = open('../data/chv-rus/chv_rus.xml','r')

soup = BeautifulSoup(data, 'xml')

train_rus= open("train.tags.chv-rus.rus","w+")
train_chv= open("train.tags.chv-rus.chv","w+")

valid_rus= open("valid.tags.chv-rus.rus","w+")
valid_chv= open("valid.tags.chv-rus.chv","w+")

test_rus= open("test.tags.chv-rus.rus","w+")
test_chv= open("test.tags.chv-rus.chv","w+")

counter = 0

for tag in soup.find_all("pair"):
      if counter % 30==0:
          test_rus.write(tag.rus.text.replace('\n',' ').strip() + '\n')
          test_chv.write(tag.chv.text.replace('\n',' ').strip() + '\n')
      elif counter % 31==0:
          valid_rus.write(tag.rus.text.replace('\n',' ').strip() + '\n')
          valid_chv.write(tag.chv.text.replace('\n',' ').strip() + '\n')
      else:
          train_rus.write(tag.rus.text.replace('\n',' ').strip() + '\n')
          train_chv.write(tag.chv.text.replace('\n',' ').strip() + '\n')
      counter+=1

    
train_rus.close()
train_chv.close()

valid_rus.close()
valid_chv.close()

test_rus.close()
test_chv.close()


from nltk.tokenize import WordPunctTokenizer
def tokenize(src_filename, new_filename):
  with open(src_filename, encoding="utf-8") as src_file:
    with open(new_filename, "w", encoding="utf-8") as new_file:
      for line in src_file:
        new_file.write("%s" % ' '.join(WordPunctTokenizer().tokenize(line)))
        new_file.write("\n")

#train
tokenize("train.tags.chv-rus.rus", "train.tok.chv-rus.rus")
tokenize("train.tags.chv-rus.chv", "train.tok.chv-rus.chv")
#test
tokenize("test.tags.chv-rus.rus", "test.tok.chv-rus.rus")
tokenize("test.tags.chv-rus.chv", "test.tok.chv-rus.chv")
#valid
tokenize("valid.tags.chv-rus.rus", "valid.tok.chv-rus.rus")
tokenize("valid.tags.chv-rus.chv", "valid.tok.chv-rus.chv")

