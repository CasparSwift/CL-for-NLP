import argparse
import os
import shutil


def parse_train_args():
    parser = argparse.ArgumentParser("Continual Sequence Tagging")

    parser.add_argument("--adam_epsilon", type=float, default=1e-8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--replay_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--n_labels", type=int, default=5)
    #parser.add_argument("--n_test", type=int, default=7600)
    #parser.add_argument("--n_train", type=int, default=115000)
    parser.add_argument("--ratio", type=float, default=1.0)
    parser.add_argument("--n_workers", type=int, default=6)
    parser.add_argument("--output_dir", type=str, default="output0")
    parser.add_argument("--overwrite", action="store_true", default=True)
    parser.add_argument("--tasks", nargs='+', default=['CoNLL2003', 'OntoNotes'])
    parser.add_argument("--task_type", type=str, default="NER")
    parser.add_argument("--vocab_path", type=str, default="./data/embed/vocab.txt")
    parser.add_argument("--embed_path", type=str, default="./data/embed/embeddings.npy")
    parser.add_argument("--max_len", type=int, default=500)
    parser.add_argument("--max_doc_len", type=int, default=100)

    args = parser.parse_args()

    if args.debug:
        #args.n_train = 10000
        args.logging_steps = 1
        #args.n_test = 1000
        args.output_dir = "output_debug"
        args.overwrite = True
        args.ratio = 0.02

    if os.path.exists(args.output_dir):
        if args.overwrite:
            choice = 'y'
        else:
            choice = input("Output directory ({}) exists! Remove? ".format(args.output_dir))
        if choice.lower()[0] == 'y':
            shutil.rmtree(args.output_dir)
            os.makedirs(args.output_dir)
        else:
            raise ValueError("Output directory exists!")
    else:
        os.makedirs(args.output_dir)
    return args