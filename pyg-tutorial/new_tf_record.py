import functools
import json
import reading_utils
import tensorflow as tf
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def _read_metadata():
    with open('/Users/daniel/Documents/Python Projects/AiPEX-Projects/pyg-tutorial/data/raw/WaterDrop/metadata.json', 'rt') as fp:
        return json.loads(fp.read()) 

metadata = _read_metadata()
ds = tf.data.TFRecordDataset(["/Users/daniel/Documents/Python Projects/AiPEX-Projects/pyg-tutorial/data/raw/WaterDrop/test.tfrecord"])
print (ds) ## Dictionary Structure

ds = ds.map(functools.partial(reading_utils.parse_serialized_simulation_example, metadata=metadata))
print(ds)
lds = list(ds)

### Your dataset comprises of 30 simulations, each simulation contains 1001 timesteps
### and a variable number of particles. 
# The first index in lds corresponds to simulation number, don't change the second index.
# You can access each of the 30 simulations by only changing the 1st index. 
ptypes = lds[0][0]['particle_type'].numpy()
positions = lds[0][1]['position'].numpy()
print (ptypes.shape, positions.shape)


