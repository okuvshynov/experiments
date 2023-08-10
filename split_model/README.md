

### test loading llama2_7b

```
python split_model/manual_load.py ../llama-2-7b/consolidated.00.pth ../llama-2-7b/params.json
```

### check memory usage

```
python split_model/track_ram.py ../llama-2-7b/consolidated.00.pth ../llama-2-7b/params.json
```

### TODO:
[ ] just path to llama folder, no individual files
[ ] make backprop work. Have to use larger device to test, no way to run locally
[ ] test on large fast machine
[ ] better handling of device
[ ] some progress + measure ram usage internally