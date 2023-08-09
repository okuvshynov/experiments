import pickle
import sys
import collections
import torch

def load_torch_pickle(fpath):
    class DummyObj:
        def __init__(self, *args):
            pass

    class DummyUnpickler(pickle.Unpickler):
        def find_class(self, module, name):
            print(f'custom_find_class {module} {name}')
            if name == 'OrderedDict':
                return collections.OrderedDict
            return DummyObj

    with open(fpath, 'rb') as f:
        def persistent_load(saved_id):
            #print(f'persistent_load {saved_id}')
            typename, storage_type, key, location, numel = saved_id
            return None
        
        unpickler = DummyUnpickler(f)

        unpickler.persistent_load = persistent_load

        result = unpickler.load()
        return list(result.keys())


keys = load_torch_pickle(sys.argv[1])
print(keys)

model_list = [(k, t.shape) for k, t in torch.load(sys.argv[2]).items()]
print(model_list)