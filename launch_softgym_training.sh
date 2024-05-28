conda activate mtm-softgym
source ~/softgym/prepare_1.0.sh
export PYTHONPATH=$PYTHOPATH:$PWD
python ./research/mtm/train_softgym.py
