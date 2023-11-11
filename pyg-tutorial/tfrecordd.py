# import collections
import networkx as nx
from collections.abc import Iterable
import collections
collections.Iterable = Iterable
from sklearn.neighbors import NearestNeighbors, KDTree
from sklearn import neighbors
import functools
import json
import reading_utils
import tensorflow as tf
import tensorflow_datasets as tfds
import sys
import numpy as np
np.set_printoptions(threshold=sys.maxsize)

def _read_metadata():
    with open('data/raw/WaterDrop/metadata.json', 'rt') as fp:
        return json.loads(fp.read())

def prepare_inputs(tensor_dict):
  """Prepares a single stack of inputs by calculating inputs and targets.

  Computes n_particles_per_example, which is a tensor that contains information
  about how to partition the axis - i.e. which nodes belong to which graph.

  Adds a batch axis to `n_particles_per_example` and `step_context` so they can
  later be batched using `batch_concat`. This batch will be the same as if the
  elements had been batched via stacking.

  Note that all other tensors have a variable size particle axis,
  and in this case they will simply be concatenated along that
  axis.

  Args:
    tensor_dict: A dict of tensors containing positions, and step context (
    if available).

  Returns:
    A tuple of input features and target positions.

  """
  # Position is encoded as [sequence_length, num_particles, dim] but the model
  # expects [num_particles, sequence_length, dim].
  pos = tensor_dict['position']
  pos = tf.transpose(pos, perm=[1, 0, 2])

  # The target position is the final step of the stack of positions.
  target_position = pos[:, -1]

  # Remove the target from the input.
  tensor_dict['position'] = pos[:, :-1]

  # Compute the number of particles per example.
  num_particles = tf.shape(pos)[0]
  # Add an extra dimension for stacking via concat.
  tensor_dict['n_particles_per_example'] = num_particles[tf.newaxis]

  if 'step_context' in tensor_dict:
    # Take the input global context. We have a stack of global contexts,
    # and we take the penultimate since the final is the target.
    tensor_dict['step_context'] = tensor_dict['step_context'][-2]
    # Add an extra dimension for stacking via concat.
    tensor_dict['step_context'] = tensor_dict['step_context'][tf.newaxis]
  return tensor_dict, target_position


title = "MultiMaterial"
condition = "train"
metadata = _read_metadata()
ds = tf.data.TFRecordDataset(["data/raw/WaterRamps/train.tfrecord"])
# ds = tf.data.TFRecordDataset(["/home/aipex/Sakthi/deepmind-research/learning_to_simulate/WaterRamps/{}/{}.tfrecord".format(title,condition)])
# ds = ds.map(functools.partial(reading._utils.parse_serialized_simulation_example, metadata=metadata))
print (ds)

# ds = ds.repeat()
# ds = ds.shuffle(512)
# iterator = iter(ds)
# next_data = iterator.get_next()
# print (next(iter(ds)))

# lds = list(ds)
# ptypes = lds[0][0]['particle_type'].numpy()
# key = lds[0][0]['key'].numpy()
# positions = lds[0][1]['position'].numpy()
# print (ptypes.shape)

ds_np = tfds.as_numpy(ds)
# print(ds_np)

print(len(next(iter(ds_np))))






np.save('positions_{}_{}.npy'.format(title, condition),positions)
np.save('ptypes_{}_{}.npy'.format(title, condition),ptypes)
# positions = np.load('positions_{}_{}.npy'.format(title, condition))
# ptypes = np.load('ptypes_{}_{}.npy'.format(title, condition))

rule = lambda x: (1318 + np.random.normal(0, 1)) if x == 7 else ((1000 + np.random.normal(0, 1)) if x == 5 else (1442 + np.random.normal(0, 1)))
# rule = lambda x: (2500 + np.random.normal(0, 1)) if x == 3 else ((1000 + np.random.normal(0, 1)) if x == 5 else (1442 + np.random.normal(0, 1)))

densities = np.asarray(list(map(rule, ptypes)))

positions_cap = np.zeros((positions.shape[0]-1, positions.shape[1], 8), dtype=np.float64)
positions_cap = np.zeros((positions.shape[0]-1, positions.shape[1], 8), dtype=np.float64)

ptype_cap = np.zeros((ptypes.shape[0]-1))

# print(len(set(list(ptypes))))


for timestep in range(1, positions.shape[0]-1):
    velocity = positions[timestep, :, :] - positions[timestep-1, :, :]
    acceleration = positions[timestep+1, :, :] - positions[timestep, :, :] - velocity
    positions_cap[timestep-1] = np.concatenate([positions[timestep], velocity, acceleration, ptypes.reshape(-1, 1), densities.reshape(-1, 1)], axis=1)


print (positions_cap[0,:5,:])
print(positions_cap[:, :, :4].shape, positions_cap[:, :, 6:].shape)
data_x = np.concatenate([positions_cap[:, :, :4], positions_cap[:, :, 6:]], axis=2)
data_y = positions_cap[:, :, 4:6]
data_x = np.expand_dims(data_x, axis=0)
data_y = np.expand_dims(data_y, axis=0)
print (data_x.shape, data_y.shape)
print(data_x[0, 0, :5, :])
print(data_y[0, 0, :5, :])


np.save("{}_{}_x.npy".format(title,condition), data_x)
np.save("{}_{}_y.npy".format(title,condition), data_y)

def edgelist_maker(xb, cnt):
    graph = []
    for nodes in xb:
        for node in range(1, nodes.shape[0]):
            anchor = nodes[0]
            neighbors = [anchor, nodes[node]]
            graph.append(neighbors)
    unique_set = set(tuple(xa) for xa in graph)
    graph = [tuple(elem) for elem in unique_set]
    modified = np.asarray(graph)
    modified = set(list(modified[:, 0]))
    for j in range(xb.shape[0]):
        if j not in modified:
            graph.append(tuple((j, j)))
    G = nx.from_edgelist(graph)
    adj = nx.to_numpy_matrix(G, nodelist=sorted(G.nodes()))
    np.save("/data/processed/WaterRamps" + "/" + "{}_{}.npy".format(cnt, condition), adj)

def collision_maker(dx, cnt):

    xx = np.repeat(a=dx[0, cnt, :, :2], repeats=dx.shape[2], axis=0)
    xxx = np.tile(dx[0, cnt, :, :2], (dx.shape[2], 1))
    all = np.concatenate([xx,xxx], axis=1)
    dist = np.sqrt(np.sum(np.square(all[:, :2] - all[:, 2:4]),axis=1))
    rule = lambda x: 0 if x>0.015 else 1
    dist = np.asarray(list(map(rule, dist)))
    dist = dist.reshape(-1, dx.shape[2])
    ones = np.eye(dist.shape[0])
    Adj = dist-ones
    # np.save("/home/aipex/Sakthi/deepmind-research/learning_to_simulate/WaterRamps/{}".format(title) + "/" + "{}_{}_{}_collision.npy".format(title, cnt, condition), Adj)
    np.save("/data/processed/WaterRamps/" + "/" + "{}_{}_{}_collision.npy".format(title, cnt, condition), Adj)

for graph in range(data_x.shape[1]):
  kdt = KDTree(data_x[graph,:,:2])
  xxb = kdt.query_radius(data_x[graph, :, :2], r=0.015)
  edgelist_maker(xxb,graph)
  collision_maker(data_x, graph)




# TYPE_TO_COLOR = {
#     3: "black",  # Boundary particles.
#     0: "green",  # Rigid solids.
#     7: "magenta",  # Goop.
#     6: "gold",  # Sand.
#     5: "blue",  # Water.
# }

# <MapDataset shapes: 
# ({particle_type: (None,), key: ()}, 
# {position: (601, None, 2)}), 
# types: ({particle_type: tf.int64, key: tf.int64}, {position: tf.float32}) >
