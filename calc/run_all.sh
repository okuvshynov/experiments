# cursor
python data/prepare.py cursor
python train.py train_addition.py --masked=True
python train.py train_addition.py --masked=False

# default
python data/prepare.py
python train.py train_addition.py --masked=True
python train.py train_addition.py --masked=False

# inverted
python data/prepare.py inverted
python train.py train_addition.py --masked=True --invert_eval=True
python train.py train_addition.py --masked=False --invert_eval=True