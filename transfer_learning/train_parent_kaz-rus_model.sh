echo 'Installing Sockeye...'
pip3 install sockeye

echo 'Installing mxnet...'
pip3 install mxnet-cu100mkl

echo 'Installing mxnet...'
pip3 install nltk

echo 'Downloading Subword-NMT'
git clone https://github.com/rsennrich/subword-nmt.git

echo 'Preparing kaz-rus dataset...'
pip3 install beautifulsoup4
python3 bs_parser_kaz.py


echo 'Preparing chv-rus dataset...'
python3 bs_parser_chv.py


cp train.tok.kaz-rus.kaz kazchv.all.train.tok
cat train.tok.chv-rus.chv >> kazchv.all.train.tok
cp train.tok.kaz-rus.rus rus.all.train.tok
cat train.tok.chv-rus.rus >> rus.all.train.tok


echo 'Preparing kaz-rus BPE and Vocab'
python3  subword-nmt/subword_nmt/learn_joint_bpe_and_vocab.py --input kazchv.all.train.tok rus.all.train.tok -s 10000 -o bpe.codes --write-vocabulary bpe.vocab.kazchv bpe.vocab.rus

echo '    Apply BPE to All data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.kazchv --vocabulary-threshold 50 < kazchv.all.train.tok > kazchv.all.train.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < rus.all.train.tok > rus.all.train.bpe


echo '    Apply BPE to Train data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.kazchv --vocabulary-threshold 50 < train.tok.kaz-rus.kaz > train.kaz-rus.kaz.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < train.tok.kaz-rus.rus > train.kaz-rus.rus.bpe


echo '    Apply BPE to Test data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.kazchv --vocabulary-threshold 50 < test.tok.kaz-rus.kaz >  test.kaz-rus.kaz.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < test.tok.kaz-rus.rus >  test.kaz-rus.rus.bpe

echo '    Apply BPE to Validation data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.kazchv --vocabulary-threshold 50 < valid.tok.kaz-rus.kaz >  valid.kaz-rus.kaz.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < valid.tok.kaz-rus.rus >  valid.kaz-rus.rus.bpe


echo 'Number of Training samples'
wc -l train.kaz-rus.rus.bpe 
echo 'Number of Test samples'
wc -l test.kaz-rus.rus.bpe
echo 'Number of Validation samples'
wc -l valid.kaz-rus.rus.bpe

echo 'Prepare data for Sockeye training...'
python3 -m sockeye.prepare_data -s kazchv.all.train.bpe -t rus.all.train.bpe -o kazrus_all_data

python3 -m sockeye.prepare_data -s train.kaz-rus.kaz.bpe -t train.kaz-rus.rus.bpe -o kazrus_parent_data --source-vocab kazrus_all_data/vocab.src.0.json --target-vocab kazrus_all_data/vocab.trg.0.json


echo 'Training...'

python3 -m sockeye.train -d kazrus_all_data -vs valid.kaz-rus.kaz.bpe -vt valid.kaz-rus.rus.bpe --encoder transformer --decoder transformer --transformer-model-size 512 --transformer-feed-forward-num-hidden 256 --transformer-dropout-prepost 0.1 --num-embed 512 --max-seq-len 100 --decode-and-evaluate 500 -o kazrus_parent_model --num-layers 6 --disable-device-locking --batch-size 1024 --optimized-metric bleu --max-num-checkpoint-not-improved 10 

echo 'Training Finished...'

