# Learning General World Models in a Handful of Reward-Free Deployments

Please follow the instructions in https://github.com/danijar/dreamerv2 to set up DreamerV2.

Example usage:
```
python -m main \
--wandb_base_url=https://api.wandb.ai \
--wandb_api_key=<your wandb api key> \
--task=atari_montezuma_revenge \
--method=single_disag \
--num_agents=10 \
--seed=100 \
--train_every=2000 \
--envs=10 \
--envs_parallel=none \
--steps=10000 \
--explorer_train_steps=5000 \
--explorer_reinit=false \
--offline_model_train_steps=5000 \
--eval_eps=10 \
--eval_type=none \
--cascade_alpha=0.0 \
--logdir=~/wm_logs/ \
--wandb_project=montezuma_sweep \
--checkpoint=true \
--xpid=experiment
```
