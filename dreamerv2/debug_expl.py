
import time
start = time.time()
### this is the cascade reward from expl.py, but in a script format so it can be easily debugged
cascade_metric = "euclidean"
cascade_k = 5
cascade_proj = 0
cascade_states = "final"

from scipy.spatial.distance import cdist
## ompute cascade rewards
cascade_rew = []
for i, seq in enumerate(seqs):
    if i == 0:
        cascade_rew += [tf.cast(tf.zeros([seq['deter'].shape[0], seq['deter'].shape[1]]), tf.float32)]
    elif cascade_states == "all":
        cascade = tf.concat([seq['deter'] for seq in seqs[:i]], axis=0).numpy()
        cascade = cascade.reshape(-1, cascade.shape[-1])
        obs = seq['deter'].numpy()
        obs = obs.reshape(-1, obs.shape[-1])
        if cascade_proj > 0:
            # project to low dim
            pass
        dists = cdist(obs, cascade, metric=cascade_metric)
        cascade_rew += [tf.cast(tf.tensor(np.mean(np.partition(dists, cascade_k, axis=1)[:, :cascade_k], axis=1).reshape(seq['deter'].shape[0],seq['deter'].shape[1])), tf.float32)]
    elif cascade_states == "final":
        cascade = tf.concat([seq['deter'][-1] for seq in seqs[:i]], axis=0).numpy()
        obs = seq['deter'][-1].numpy()
        dists = cdist(obs, cascade, metric=cascade_metric)
        rew = tf.cast(tf.tensor(np.mean(np.partition(dists, cascade_k, axis=1)[:, :cascade_k], axis=1).reshape(1, seq['deter'].shape[1])), tf.float32)
        cascade_rew += [tf.concat([tf.cast(tf.zeros([seq['deter'].shape[0]-1, seq['deter'].shape[1]]), tf.float32), rew], axis=0)]

print(f"time taken = {time.time()-start}")