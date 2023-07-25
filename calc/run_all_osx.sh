# cursor
python data/prepare.py cursor
python train.py train_addition.py --masked=True --device=mps --compile=False
python train.py train_addition.py --masked=False --device=mps --compile=False

# default
python prepare.py
python train.py train_addition.py --masked=True --device=mps --compile=False
python train.py train_addition.py --masked=False --device=mps --compile=False

# inverted
python prepare.py inverted
python train.py train_addition.py --masked=True --device=mps --compile=False --invert_eval=True
python train.py train_addition.py --masked=False --device=mps --compile=False --invert_eval=True