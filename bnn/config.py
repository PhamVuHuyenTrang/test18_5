import logging
import datetime
import argparse
import sys
import os


def config():
    parser = argparse.ArgumentParser()

    # Data Provider
    parser.add_argument('--kaggle', action='store_true')
    parser.add_argument('--no-kaggle', dest='kaggle', action='store_false')
    parser.set_defaults(kaggle=False)

    parser.add_argument('--ggcolab', action='store_true')
    parser.add_argument('--no-ggcolab', dest='ggcolab', action='store_false')
    parser.set_defaults(ggcolab=False)

    parser.add_argument('--dataset', type=str, default="permutedMNIST")
    parser.add_argument('--chunk-size', type=int, default=100)
    parser.add_argument('--n-chunks', type=int, default=1000)
    parser.add_argument('--n-drifts', type=int, default=6)
    parser.add_argument('--changerate', type=int, default=3)
    parser.add_argument('--data-random-state', type=int, default=1410)
    parser.add_argument('--max-iter', type=int, default=100)

    # HypothesesStorage
    if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
        parser.add_argument(
            '--prune', action=argparse.BooleanOptionalAction, default=True)
    else:
        parser.add_argument('--prune', action='store_true')
        parser.add_argument('--no-prune', dest='prune', action='store_false')
        parser.set_defaults(prune=True)

    parser.add_argument('--num-models', type=int, default=3)
    parser.add_argument('--Lambda', type=float, default=0.5, help='prior probability of having concept drift')
    parser.add_argument('--noise', type=float, default=1.0)
    
    parser.add_argument('--variational-dropout', action='store_true')
    parser.add_argument('--no-variational-dropout', dest='variational-dropout', action='store_false')
    parser.set_defaults(variational_dropout=False)

    parser.add_argument('--hnet', action='store_true')
    parser.add_argument('--no-hnet', dest='hnet', action='store_false')
    parser.set_defaults(hnet=False)

    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--no-save-model', dest='save-model', action='store_false')
    parser.set_defaults(save_model=False)

    parser.add_argument('--temperature', type=float, default=0.01)
    parser.add_argument('--weight-kl-dropout', type=float, default=1.0)
    parser.add_argument('--weight-kl-dense', type=float, default=0.1)
    parser.add_argument('--max_init-drop-rate-factor',
                        type=float, default=1.386)
    parser.add_argument('--min_init-drop-rate-factor', type=float, default=0.)
    parser.add_argument('--droprate-init-strategy', type=int, default=0)
    parser.add_argument('--diffusion', type=float, default=1.5)
    parser.add_argument('--jump-bias', type=float, default=0.0)

    # Trainer
    parser.add_argument('--device', type=str, default='whatever')
    parser.add_argument('--mcSamples', type=int, default=1, help='make sure each step have mcSamples*batch_size samples feed into RAM')
    parser.add_argument('--mcSamples-finalround', type=int, default=1)
    parser.add_argument('--mcSamples-test', type=int, default=1)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--learning-rate', type=float, default=3e-4)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--surrogate-prior-path', type=str,
                        default='VCLDropoutCheckpoint\epoch16.pt')
    parser.add_argument('--hnet-path', type=str, default='checkpoint/MNIST_hnet/hnet_epoch94.pt')
    parser.add_argument('--emb-path', type=str, default='checkpoint/MNIST_hnet/hnet_task_embedding94.pt')
    if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
        parser.add_argument('--train-from-scratch',
                            action=argparse.BooleanOptionalAction, default=False)
    else:
        parser.add_argument('--train-from-scratch', action='store_true')
        parser.add_argument('--no-train-from-scratch',
                            dest='train-from-scratch', action='store_false')
        parser.set_defaults(multiprocessing=False)
    if sys.version_info[0] >= 3 and sys.version_info[1] >= 9:
        parser.add_argument(
            '--multiprocessing', action=argparse.BooleanOptionalAction, default=False)
    else:
        parser.add_argument('--multiprocessing', action='store_true')
        parser.add_argument('--no-multiprocessing',
                            dest='multiprocessing', action='store_false')
        parser.set_defaults(multiprocessing=False)
    parser.add_argument('--cpus', type=int, default=4,
                        help='should = num-models + 1')

    config = parser.parse_args()
    if config.kaggle and config.ggcolab:
        raise ValueError('kaggle and ggcolab cannot be both choosen')


    if not os.path.exists("log/"):
        os.makedirs("log")
    formatter = logging.Formatter('%(message)s')

    dropoutLogger = logging.getLogger("dropoutLogger")
    dropoutLogger.setLevel(logging.INFO)

    file_name = 'log/dropoutlogging.log'
    dropout_file_handler = logging.FileHandler(file_name)
    dropout_file_handler.setFormatter((formatter))
    dropoutLogger.addHandler(dropout_file_handler)

    logger = logging.getLogger("logger")
    logger.setLevel(logging.INFO)

    file_name = 'log/' + \
        f'Dataset-{config.dataset}-dropout-{config.variational_dropout}-numModels-{config.num_models}-KLdense-diffusion{config.diffusion}-jumpbias-{config.jump_bias}-{config.weight_kl_dense}-KLdrop-{config.weight_kl_dropout}-Lambda-{config.Lambda}-lr-{config.learning_rate}' \
        + '.log'


    file_handler = logging.FileHandler(file_name)
    # file_handler = logging.StreamHandler(sys.stdout)
    file_handler.setFormatter((formatter))

    logger.addHandler(file_handler)

    logger = logging.getLogger('logger')
    logger.info(f'TIME: {datetime.datetime.now()}')
    logger.info(
        f'DATA: {config.n_chunks} timestep - chunk size {config.chunk_size} - {config.n_drifts} drifts - random state: {config.data_random_state}')
    logger.info(f'MODEL: Lambda = {config.Lambda}  - jumpbias = {config.jump_bias}- prune: {config.prune} - {config.num_models} models - mcSamples: {config.mcSamples} \n\t\t weight-kl-drop: {config.weight_kl_dropout} - weight-kl-dense: {config.weight_kl_dense} - max_init_drop_rate_factor: {config.max_init_drop_rate_factor} - min_init_drop_rate_factor: {config.min_init_drop_rate_factor}')
    logger.info(
        f'TRAINER: multiprocessing: {config.multiprocessing} - {config.cpus} cpus - lr: {config.learning_rate} - {config.epochs} epochs - device: {config.device} - batch-size: {config.batch_size}')
    logger.info(f'\n')
    return config
