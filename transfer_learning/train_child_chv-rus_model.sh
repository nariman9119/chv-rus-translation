

echo '    Apply BPE to Train data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.kazchv --vocabulary-threshold 50 < train.tok.chv-rus.chv > train.chv-rus.chv.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < train.tok.chv-rus.rus > train.chv-rus.rus.bpe


echo '    Apply BPE to Test data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.kazchv --vocabulary-threshold 50 < test.tok.chv-rus.chv > test.chv-rus.chv.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < test.tok.chv-rus.rus > test.chv-rus.rus.bpe

echo '    Apply BPE to Validation data'
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.kazchv --vocabulary-threshold 50 < valid.tok.chv-rus.chv > valid.chv-rus.chv.bpe
python3 subword-nmt/subword_nmt/apply_bpe.py -c bpe.codes --vocabulary bpe.vocab.rus --vocabulary-threshold 50 < valid.tok.chv-rus.rus > valid.chv-rus.rus.bpe


echo 'Number of Training samples'
wc -l train.chv-rus.chv.bpe
echo 'Number of Test samples'
wc -l test.chv-rus.chv.bpe
echo 'Number of Validation samples'
wc -l valid.chv-rus.chv.bpe

echo 'Prepare data for Sockeye training...'

python3 -m sockeye.prepare_data -s train.chv-rus.chv.bpe  -t train.chv-rus.rus.bpe -o chvrus_child_data  --source-vocab kazrus_all_data/vocab.src.0.json --target-vocab kazrus_all_data/vocab.trg.0.json
echo 'Training...'

python3 -m sockeye.train -d chvrus_child_data -vs valid.chv-rus.chv.bpe -vt valid.chv-rus.rus.bpe --encoder transformer --decoder transformer --transformer-model-size 512 --transformer-feed-forward-num-hidden 256 --transformer-dropout-prepost 0.1 --num-embed 512 --max-seq-len 100 --decode-and-evaluate 500 -o chvrus_parent_model --num-layers 6 --disable-device-locking --batch-size 1024 --optimized-metric bleu --max-num-checkpoint-not-improved 10 



echo 'Training Finished...'
