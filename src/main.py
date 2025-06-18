import click
from runs import RunFactory
import os


@click.command(context_settings=dict(
    ignore_unknown_options=True,
))
@click.option('--run', '-r', default='compas_fairlab', help='Run to execute')
@click.option('--project_name', '-p', default='CompasFairLab', help='Project name')
@click.option('--id', '-i', default='test', help='Run id')
@click.option('--num_local_iterations', '-nl', default=30, help='Number of local iterations')
@click.option('--global_patience', '-gp', default=5, help='Global patience')
@click.option('--local_patience', '-lp', default=5, help='Client local patience')
@click.option('--num_clients', '-nc', default=1, help='Number of clients')
@click.option('--gpu_devices', '-g', multiple=True, help='List of GPU devices')
@click.option('--num_federated_iterations', '-nf', default=30, help='Number of federated (server) iterations')
@click.option('--experiment_name', '-e', default='Dirichlet_09', help='Experiment name')
@click.option('--num_classes', default=2, help='Number of classes in the dataset')
def main(run, project_name, id,
         num_local_iterations,
         global_patience, local_patience,
         num_clients, gpu_devices,
         num_federated_iterations,
         experiment_name,
         num_classes
         ):

    if gpu_devices:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_devices)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    run = RunFactory.create_run(run,
                                project_name=project_name,
                                id=id,
                                num_global_iterations=1,
                                num_local_iterations=num_local_iterations,
                                global_patience=global_patience,
                                local_patience=local_patience,
                                num_clients=num_clients,
                                num_federated_iterations=num_federated_iterations,
                                experiment_name=experiment_name,
                                num_classes=num_classes
                                )
    run()


if __name__ == '__main__':
    # mp.set_start_method("spawn", force=True)
    main()
