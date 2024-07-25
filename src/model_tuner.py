from pytorch_lightning.loggers import NeptuneLogger
from ray import tune, train
from ray.tune import CLIReporter, JupyterNotebookReporter
from ray.tune.schedulers import ASHAScheduler
import classifier
import loader

def tune_classifier(num_samples=10, num_epochs=10, gpus_per_trial=1, logger_config=None, dm=None):
    # param space
    config = {
        "layers_count": tune.choice([32, 64, 128]),
        "kernel_filter": tune.choice([64, 128, 256]),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.choice([32]),
        "num_epochs": num_epochs,
    }

    if dm is None:
        dm = loader.AudioDataModule(batch_size=config['batch_size'])

    trainable = tune.with_parameters(
        classifier.train_func,
        num_epochs=num_epochs,
        logger_config=logger_config,
        dm=dm,
    )

    # scheduler = ASHAScheduler(max_t=num_epochs, grace_period=1, reduction_factor=2)

    reporter = JupyterNotebookReporter(
        parameter_columns=["layers_count", "kernel_filter", "lr", "batch_size"],
        metric_columns=["loss", "acc"],
    )

    tuner = tune.Tuner(
        tune.with_resources(trainable, resources={"cpu": 1, "gpu": gpus_per_trial}),
        tune_config=tune.TuneConfig(
            metric="loss",
            mode="min",
            # scheduler=scheduler,
            num_samples=num_samples,
        ),
        run_config=train.RunConfig(
            name="tune_digits_classifier",
            progress_reporter=reporter,
        ),
        param_space=config,
    )
    results = tuner.fit()

    print("Best hyperparameters found were: ", results.get_best_result().config)