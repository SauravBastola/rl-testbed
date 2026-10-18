"""Choose the training loop for an environment.

Language tasks are trained by SequenceTrainer (batches of generated answers);
everything else by the step-by-step Trainer. Both take the same arguments and
write the same files.
"""


def create_trainer(**kwargs):
    from rltestbed.core.envs import REGISTRY

    spec = REGISTRY[kwargs["env_name"]]["spec"]
    if spec.category == "language":
        from rltestbed.core.lm.trainer import SequenceTrainer

        return SequenceTrainer(**kwargs)
    from rltestbed.core.trainer import Trainer

    return Trainer(**kwargs)
