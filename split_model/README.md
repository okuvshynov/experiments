# test prefetched model
```
python split_model/split_model.py
```

# test loading llama2_7b

```
python split_model/manual_load.py ../llama-2-7b/consolidated.00.pth ../llama-2-7b/params.json
```

# check memory usage

```
python split_model/track_ram.py ../llama-2-7b/consolidated.00.pth ../llama-2-7b/params.json
```