import numpy as np
import os

from torch.utils.data import DataLoader

from book_corpus import (BookCorpusMultiProcessor, BookCorpusDataset,
                         BatchingDataset)
from utils import Config, StopWatch
from vocab import Vocab


def prepare_data_and_vocab(cfg):
    load_data_file = ["books_large_p1.txt",
                      "books_large_p2.txt"]
    #load_data_file = "books_100k.txt"

    save_data_path = os.path.join(cfg.prepro_dir, "data")
    save_vocab_path = os.path.join(cfg.prepro_dir, "vocab")

    StopWatch.go('Total')
    if (not os.path.exists(save_data_path)
        or not os.path.exists(save_vocab_path)
        or cfg.is_reload_prepro):

        print('Start preprocessing data and building vocabulary!')
        if isinstance(load_data_file, (list, tuple)):
            load_data_path = [*map(lambda fn: os.path.join(cfg.data_dir, fn),
                                   load_data_file)]
            book_procs = BookCorpusMultiProcessor.from_multiple_files(
                    file_paths=load_data_path,
                    min_len=cfg.prepro_min_len,
                    max_len=cfg.prepro_max_len)
            sents, counter = BookCorpusMultiProcessor.multi_process(book_procs)
        else:
            load_data_path = os.path.join(cfg.data_dir, load_data_file)
            book_procs = BookCorpusMultiProcessor(file_path=load_data_path,
                                                  min_len=cfg.prepro_min_len,
                                                  max_len=cfg.prepro_max_len)
            sents, counter = book_procs.process()

        vocab = Vocab(counter=counter,
                      max_size=cfg.vocab_size,
                      embed_dim=cfg.embed_dim,
                      is_load_glove=cfg.is_load_glove)
        sents = vocab.numericalize_sents(sents)

        if not os.path.exists(cfg.prepro_dir):
            os.makedirs(cfg.prepro_dir)

        with StopWatch('Saving text'):
            np.savetxt(save_data_path, sents, fmt="%s")
        with StopWatch('Pickling vocab'):
            vocab.pickle(save_vocab_path)
    else:
        print('Previously processed files will be used!')
        vocab = Vocab.unpickle(save_vocab_path)
    StopWatch.stop('Total')
    return vocab


if __name__ == '__main__':
    configuration = dict(prepro_dir="prepro_large",
                         data_dir="data/books",
                         is_reload_prepro=False,
                         is_load_glove=True,
                         batch_size=32,
                         prepro_min_len=10,
                         prepro_max_len=20,
                         embed_dim=50,
                         vocab_size=10000)
    cfg = Config(configuration)
    vocab = prepare_data_and_vocab(cfg)

    save_data_path = os.path.join(cfg.prepro_dir, "data")
    book_corpus = BookCorpusDataset(save_data_path)
    batching_dataset = BatchingDataset(vocab)
    data_loader = DataLoader(book_corpus, cfg.batch_size, shuffle=True,
                             num_workers=4, collate_fn=batching_dataset)
