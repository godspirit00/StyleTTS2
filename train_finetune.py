"""Fine-tune a pre-trained (second-stage) StyleTTS2 model on a new dataset.

The training loop is shared with train_second.py (see ``train_second.run``);
``finetune=True`` selects the fine-tuning behaviour of the original script.
Multi-GPU uses DataParallel, like train_second.py.
"""
import click

from train_second import run


@click.command()
@click.option('-p', '--config_path', default='Configs/config_ft.yml', type=str)
def main(config_path):
    run(config_path, finetune=True)


if __name__ == "__main__":
    main()
