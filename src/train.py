"""Train an EEG-to-Text decoding model with Curriculum Semantic-aware Contrastive Learning."""

from utils.set_seed import set_seed


def main():
    cfg = {
        'seed': 312
         }
    set_seed(cfg['seed'])


if __name__ == "__main__":
    main()
