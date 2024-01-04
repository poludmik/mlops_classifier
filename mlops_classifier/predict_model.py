import numpy as np
import torch
import click

from visualizations.visualize import viz_main


@click.group()
def cli():
    """Command line interface."""
    pass


def load_images(image_path):
    if image_path.endswith(".npy"):
        images = np.load(image_path)
        images = torch.from_numpy(images).float()
    else:
        print("Image format not supported (please use .npy)")
    return images


def classify(model: torch.nn.Module, dataloader: torch.utils.data.DataLoader) -> None:
    """Run prediction for a given model and dataloader.

    Args:
        model: model to use for prediction
        dataloader: dataloader with batches

    Returns
        Tensor of shape [N, d] where N is the number of samples and d is the output dimension of the model

    """
    return torch.cat([model(batch) for batch in dataloader], 0)


@click.command()
@click.argument("model_pt")
@click.argument("images")
def predict(model_pt, images):

    print("Testing cuda in a docker container: ", torch.cuda.is_available())
    model = torch.load(model_pt)
    images = load_images(images)
    dataloader = torch.utils.data.DataLoader(images, batch_size=64)

    predictions = classify(model, dataloader)
    print(predictions)


@click.command()
def vizualize():
    viz_main()


cli.add_command(predict)
cli.add_command(vizualize)

if __name__ == "__main__":
    cli()
