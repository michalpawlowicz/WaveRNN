from utils.dataset import get_vocoder_datasets
from utils.dsp import *
from models.fatchord_version import WaveRNN
from utils.paths import Paths
from utils.display import simple_table
import torch
import argparse
from pathlib import Path
import time
import os


def gen_from_file(model: WaveRNN, load_path: Path, save_path: Path, batched, target, overlap):
    suffix = load_path.suffix
    if suffix == ".wav":
        wav = load_wav(load_path)
        save_wav(wav, os.path.join(save_path, "target", os.path.basename(load_path)))
        print("Generating from {0}".format(load_path))
        mel = melspectrogram(wav)
        print("Melspectrograms generated!")
    elif suffix == ".npy":
        mel = np.load(load_path)
        if mel.ndim != 2 or mel.shape[0] != hp.num_mels:
            raise ValueError(f'Expected a numpy array shaped (n_mels, n_hops), but got {wav.shape}!')
        _max = np.max(mel)
        _min = np.min(mel)
        if _max >= 1.01 or _min <= -0.01:
            raise ValueError(f'Expected spectrogram range in [0,1] but was instead [{_min}, {_max}]')
    else:
        raise ValueError(f"Expected an extension of .wav or .npy, but got {suffix}!")


    mel = torch.tensor(mel).unsqueeze(0)

    batch_str = f'gen_batched_target{target}_overlap{overlap}' if batched else 'gen_NOT_BATCHED'
    save_str = os.path.join(save_path, os.path.basename(load_path))

    beg = time.time()
    print("Start generating... [{0}]".format(beg))
    output = model.generate(mel, save_str, batched, target, overlap, hp.mu_law)
    end = time.time()
    print("Done generating... [{0}] -> delta: [{1}]".format(end, end-beg))
    save_wav(output, save_str)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate WaveRNN Samples')
    parser.add_argument('--batched', '-b', dest='batched', action='store_true', help='Fast Batched Generation')
    parser.add_argument('--output_dir', dest='output_dir', default=None, help='Output')
    parser.add_argument('--unbatched', '-u', dest='batched', action='store_false', help='Slow Unbatched Generation')
    parser.add_argument('--target', '-t', type=int, help='[int] number of samples in each batch index')
    parser.add_argument('--overlap', '-o', type=int, help='[int] number of crossover samples')
    parser.add_argument('--file', '-f', type=str, required=True, help='[string/path] for testing a wav outside dataset')
    parser.add_argument('--voc_weights', '-w', type=str, help='[string/path] Load in different WaveRNN weights')
    parser.add_argument('--gta', '-g', dest='gta', action='store_true', help='Generate from GTA testset')
    parser.add_argument('--force_cpu', '-c', action='store_true', help='Forces CPU-only training, even when in CUDA capable environment')
    parser.add_argument('--hp_file', metavar='FILE', default='hparams.py', help='The file to use for the hyperparameters')

    parser.set_defaults(batched=None)

    args = parser.parse_args()

    hp.configure(args.hp_file)  # Load hparams from file
    # set defaults for any arguments that depend on hparams
    if args.target is None:
        args.target = hp.voc_target
    if args.overlap is None:
        args.overlap = hp.voc_overlap
    if args.batched is None:
        args.batched = hp.voc_gen_batched

    batched = args.batched
    target = args.target
    overlap = args.overlap
    file = args.file
    gta = args.gta

    if not args.force_cpu and torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print('Using device:', device)

    print('\nInitialising Model...\n')

    model = WaveRNN(rnn_dims=hp.voc_rnn_dims,
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

    paths = Paths(hp.data_path, hp.voc_model_id, hp.tts_model_id)
    if args.output_dir is not None:
        paths.voc_output = args.output_dir

    voc_weights = args.voc_weights if args.voc_weights else paths.voc_latest_weights

    model.load(voc_weights)

    simple_table([('Generation Mode', 'Batched' if batched else 'Unbatched'),
                  ('Target Samples', target if batched else 'N/A'),
                  ('Overlap Samples', overlap if batched else 'N/A')])

    if os.path.isfile(file):
        file = Path(file).expanduser()
        gen_from_file(model, file, paths.voc_output, batched, target, overlap)
    else:
        files = [p.path for p in os.scandir(file)]
        for infile in files:
            if infile.endswith(".wav"):
                infile = Path(infile).expanduser()
                gen_from_file(model, infile, paths.voc_output, batched, target, overlap)

    print('\n\nExiting...\n')
