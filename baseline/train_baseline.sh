echo 'Installing Sockeye...'
pip3 install sockeye

echo 'Installing mxnet...'
pip3 install mxnet-cu100mkl

echo 'Installing mxnet...'
pip3 install nltk

echo 'Downloading Subword-NMT'
git clone https://github.com/rsennrich/subword-nmt.git

echo 'Preparing dataset...'
pip3 install beautifulsoup4
python3 bs_parser_baseline.py

echo 'Preparing BPE and Vocab'
python3  subword-nmt/subword_nmt/learn_joint_bpe_and_vocab.py --input train.tok.chv-rus.chv train.tok.chv-rus.rus -s 10000 -o bpe.codes --write-vocabulary bpe.vocab.chv bpe.vocab.rus

echo '    Apply BPE to Train data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.chv --vocabulary-threshold 50 < train.tok.chv-rus.chv > chv.all.train.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < train.tok.chv-rus.rus > rus.all.train.bpe


echo '    Apply BPE to Test data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.chv --vocabulary-threshold 50 < test.tok.chv-rus.chv > chv.all.test.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < test.tok.chv-rus.rus > rus.all.test.bpe

echo '    Apply BPE to Validation data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.chv --vocabulary-threshold 50 < valid.tok.chv-rus.chv > chv.all.valid.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < valid.tok.chv-rus.rus > rus.all.valid.bpe


echo 'Number of Training samples'
wc -l chv.all.train.bpe 
echo 'Number of Test samples'
wc -l chv.all.test.bpe
echo 'Number of Validation samples'
wc -l chv.all.valid.bpe

echo 'Prepare data for Sockeye training...'

python3 -m sockeye.prepare_data -s chv.all.train.bpe -t rus.all.train.bpe -o chvrus_all_data

echo 'Training...'

python3 -m sockeye.train -d chvrus_all_data -vs chv.all.valid.bpe -vt rus.all.valid.bpe --encoder transformer --decoder transformer --transformer-model-size 512 --transformer-feed-forward-num-hidden 256 --transformer-dropout-prepost 0.1 --num-embed 512 --max-seq-len 100 --decode-and-evaluate 500 -o chvrus_parent_model --num-layers 6 --disable-device-locking --batch-size 1024 --optimized-metric bleu --max-num-checkpoint-not-improved 10 



echo 'Training Finished...'
