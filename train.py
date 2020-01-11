
import argparse
import yaml

import models
import trainers

def get_trainer(model_type: str, params: dict):
    if model_type == "cyclegan":
        trainer = trainers.CycleGANTrainer(params)
    elif model_type == "munit":
        trainer = trainers.MUNITTrainer(params)
    else:
        raise NotImplementedError

    return trainer


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Training Image2Image translation model')
    parser.add_argument('param_file', help="Path to training parameter file (yaml file)")

    args = parser.parse_args()
    
    with open(args.param_file, "r") as f:
        params = yaml.load(f, Loader=yaml.SafeLoader)

    model_type = params["model_type"]
    assert model_type in ["cyclegan", "munit"], "model_type is not appropriate."

    domain_X = params["domain_X"]
    domain_Y = params["domain_Y"]

    trainer = get_trainer(model_type, params)

    trainer.train()