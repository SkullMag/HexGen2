import argparse

def get_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--model-size', type=str, default='llama-7b',
                        help='Instruct which model to work on')
    parser.add_argument('--num_clusters', type=int, default=None,
                        help="Instruct how many clusters to be created")
    parser.add_argument('--inter_bw', type=float, default=5,
                        help='Assumed inter-machine bandwidth')
    parser.add_argument('--machine_config_path', type=str, default=None, 
                        help='The machines to be used')
    parser.add_argument('--log_interval', type=int, 
                        help='Log interval')
    parser.add_argument('--niter', type=int, 
                        help='niter')
    parser.add_argument('--niter-maxflow', type=int, default=1,
                        help='niter-maxflow')
    parser.add_argument("--seq_in", type=int, default=128, help="input sequence length")
    parser.add_argument("--seq_out", type=int, default=64, help="output sequence length")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size")
    parser.add_argument("--visualize", action="store_true", default=False, help="Whether show the maxflow graph")
    parser.add_argument("--no-maxflow", action="store_true", help="Whether run maxflow detection")
    parser.add_argument("--use-genetic", action="store_true", help="Whether run genetic algorithm")

    args = parser.parse_args()

    return args

