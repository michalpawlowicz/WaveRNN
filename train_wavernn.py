import time
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from utils.display import stream, simple_table
from utils.dataset import get_vocoder_datasets, get_vocoder_test_dataset
from utils.distribution import discretized_mix_logistic_loss
from utils import hparams as hp
from models.fatchord_version import WaveRNN
from gen_wavernn import gen_testset
from utils.paths import Paths
import argparse
from utils import data_parallel_workaround
from utils.checkpoints import save_checkpoint, restore_checkpoint
from torch.utils.tensorboard import SummaryWriter

EPOCH=1000

def main():

    # Parse Arguments
    parser = argparse.ArgumentParser(description='Train WaveRNN Vocoder')
    parser.add_argument('--epoch', '-e', action='store', dest='epoch', default=0, help='Staring epoch value')
    parser.add_argument('--checkpoint_name', '-n', dest='checkpoint_name', default=None, help='Staring checkpoint name')
    parser.add_argument('--lr', '-l', type=float,  help='[float] override hparams.py learning rate')
    parser.add_argument('--batch_size', '-b', type=int, help='[int] override hparams.py batch size')
    parser.add_argument('--force_train', '-f', action='store_true', help='Forces the model to train past total steps')
    parser.add_argument('--gta', '-g', action='store_true', help='train wavernn on GTA features')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')
    args = parser.parse_args()

    hp.configure(args.hp_file)  # load hparams from file
    if args.lr is None:
        args.lr = hp.voc_lr
    if args.batch_size is None:
        args.batch_size = hp.voc_batch_size

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)

    batch_size = args.batch_size
    force_train = args.force_train
    train_gta = args.gta
    lr = args.lr
    EPOCH=args.epoch
    checkpoint_name=args.checkpoint_name

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
        if batch_size % torch.cuda.device_count() != 0:
            raise ValueError('`batch_size` must be evenly divisible by n_gpus!')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Model...\n')

    # Instantiate WaveRNN Model
    voc_model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
                        fc_dims=hp.voc_fc_dims,
                        bits=hp.bits,
                        pad=hp.voc_pad,
                        upsample_factors=hp.voc_upsample_factors,
                        feat_dims=hp.num_mels,
                        compute_dims=hp.voc_compute_dims,
                        res_out_dims=hp.voc_res_out_dims,
                        res_blocks=hp.voc_res_blocks,
                        hop_length=hp.hop_length,
                        sample_rate=hp.sample_rate,
                        mode=hp.voc_mode).to(device)

    # Check to make sure the hop length is correctly factorised
    assert np.cumprod(hp.voc_upsample_factors)[-1] == hp.hop_length

    optimizer = optim.Adam(voc_model.parameters())

    if checkpoint_name is not None:
        restore_checkpoint('voc', paths, voc_model, optimizer, name=checkpoint_name, create_if_missing=False) 

    train_set, test_set = get_vocoder_datasets(paths.data, batch_size, train_gta)
    test_set = get_vocoder_test_dataset(paths.data, batch_size)

    total_steps = 10_000_000 if force_train else hp.voc_total_steps

    simple_table([('Remaining', str((total_steps - voc_model.get_step())//1000) + 'k Steps'),
                  ('Batch Size', batch_size),
                  ('LR', lr),
                  ('Sequence Len', hp.voc_seq_len),
                  ('GTA Train', train_gta)])

    loss_func = F.cross_entropy if voc_model.mode == 'RAW' else discretized_mix_logistic_loss

    voc_train_loop(paths, voc_model, loss_func, optimizer, train_set, test_set, lr, total_steps)

    print('Training Complete.')
    print('To continue training increase voc_total_steps in hparams.py or use --force_train')


def voc_train_loop(paths: Paths, model: WaveRNN, loss_func, optimizer, train_set, test_set, lr, total_steps):
    # Use same device as model parameters
    device = next(model.parameters()).device

    for g in optimizer.param_groups: g['lr'] = lr

    total_iters = len(train_set)
    epochs = (total_steps - model.get_step()) // total_iters + 1

    total_number_of_batches = len(train_set)

    writer = SummaryWriter()

    for e in range(EPOCH, epochs + 1):

        start = time.time()
        running_loss = 0.
        avg_loss = 0

        for i, (x, y, m) in enumerate(train_set, 1):
            x, m, y = x.to(device), m.to(device), y.to(device)

            # Parallelize model onto GPUS using workaround due to python bug
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                y_hat = data_parallel_workaround(model, x, m)
            else:
                y_hat = model(x, m)

            if model.mode == 'RAW':
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)

            elif model.mode == 'MOL':
                y = y.float()

            y = y.unsqueeze(-1)


            loss = loss_func(y_hat, y)

            optimizer.zero_grad()
            loss.backward()
            if hp.voc_clip_grad_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hp.voc_clip_grad_norm)
                #if np.isnan(grad_norm):
                #    print('grad_norm was NaN!')
            optimizer.step()

            running_loss += loss.item()
            avg_loss = running_loss / i

            speed = i / (time.time() - start)

            step = model.get_step()
            k = step // 1000

            # Write to tensorboard per batch
            writer.add_scalar('Epoch loss', loss.item(), e*total_number_of_batches+i)

            """
            if step % hp.voc_checkpoint_every == 0:
                gen_testset(model, test_set, hp.voc_gen_at_checkpoint, hp.voc_gen_batched,
                            hp.voc_target, hp.voc_overlap, paths.voc_output)
                ckpt_name = f'wave_step{k}K'
                save_checkpoint('voc', paths, model, optimizer,
                                name=ckpt_name, is_silent=True)
            """


            msg = f'| Epoch: {e}/{epochs} ({i}/{total_iters}) | Loss: {avg_loss:.4f} | {speed:.1f} steps/s | Step: {k}k | '
            stream(msg)

        ####################### Testing ############################
        loss_test = 0
        for _, (x_test, y_test, m_test) in enumerate(test_set, 1):
            x_test, m_test, y_test = x_test.to(device), m_test.to(device), y_test.to(device)
            if device.type == 'cuda' and torch.cuda.device_count() > 1:
                raise RuntimeError("Unsupported")
            else:
                y_test_hat = model(x_test, m_test)

            if model.mode == 'RAW':
                y_test_hat = y_test_hat.transpose(1, 2).unsqueeze(-1)
            elif model.mode == 'MOL':
                y_test = y_test.float()

            y_test = y_test.unsqueeze(-1)

            loss_test += loss_func(y_test_hat, y_test).item()
        avg_loss_test = loss_test / len(test_set)
        msg = f'| Epoch: {e}/{epochs} | Test-Loss: {loss_test:.4f} | Test-AvgLoss: {avg_loss_test:.4f} | '
        stream("\n")
        stream(msg)
        ############################################################

        # Write to tensorboard per epoch
        writer.add_scalar('Running loss', running_loss, e)
        writer.add_scalar('Average loss', avg_loss, e)
        writer.add_scalar('Test loss', loss_test, e)
        writer.add_scalar('Average test loss', avg_loss_test, e)

        # Must save latest optimizer state to ensure that resuming training
        # doesn't produce artifacts
        save_checkpoint('voc', paths, model, optimizer, name="model-epoch-{0}-loss-{1}".format(e, avg_loss), is_silent=True)
        model.log(paths.voc_log, msg)
        print(' ')


if __name__ == "__main__":
    main()
