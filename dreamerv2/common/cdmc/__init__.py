
from .walker import make_walker
from .cheetah import make_cheetah

def make_dmc_all(domain, task,
         task_kwargs=None,
         environment_kwargs=None,
         visualize_reward=False):

    if domain == 'walker':
        return make_walker(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward)
    elif domain == 'cheetah':
        return make_cheetah(task,
                           task_kwargs=task_kwargs,
                           environment_kwargs=environment_kwargs,
                           visualize_reward=visualize_reward) 


