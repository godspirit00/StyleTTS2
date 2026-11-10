"""Kept for compatibility with

    accelerate launch --mixed_precision=fp16 --num_processes=1 train_finetune_accelerate.py --config_path ./Configs/config_ft.yml

It runs exactly the same code as train_finetune.py.  Mixed precision is taken
from the config's ``mixed_precision`` or, when that is 'no', from the
``--mixed_precision`` flag of ``accelerate launch``.  (The old Accelerate
version never applied autocast to the models - they were a plain dict that
``accelerator.prepare`` passed through - so "fp16" only scaled the losses.)
"""
import os
import click

from train_second import run


@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
def main(config_path):
    if int(os.environ.get('WORLD_SIZE', '1')) > 1:
        raise SystemExit('train_finetune_accelerate.py supports a single process only '
                         '(use --num_processes=1; multiple GPUs are used through '
                         'DataParallel automatically).')
    run(config_path, finetune=True)


if __name__ == "__main__":
    main()
