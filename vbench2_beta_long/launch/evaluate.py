from vbench.distributed import dist_init
from vbench2_beta_long.eval_long import main as eval_long_main


def main():
    dist_init()
    eval_long_main()


if __name__ == "__main__":
    main()
