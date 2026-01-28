import argparse
import subprocess  # noqa: S404
from pathlib import Path

from pydantic import BaseModel


class ClusterConfig(BaseModel):
    nodes: list[str]  # The first node is considered the master node
    master_port: int = 56789
    gpus_per_node: int = 1

    project_dir: Path = Path.cwd()
    torchrun_path: Path = Path(".venv/bin/torchrun")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a script on a cluster using torchrun.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="path to the cluster configuration (JSON)"
    )
    parser.add_argument(
        "-n",
        "--num-runs",
        type=int,
        default=1,
        metavar="N",
        help="number of times to run the script (default: %(default)s)",
    )

    parser.add_argument("script", type=str, help="path to the script to run")
    parser.add_argument(
        "script_args", nargs=argparse.REMAINDER, help="arguments for the run script"
    )

    return parser.parse_args()


def main() -> None:
    """Run a script on a cluster using torchrun.

    See the following links for more details:
    https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_1.html
    https://lightning.ai/docs/pytorch/stable/clouds/cluster_intermediate_2.html
    """
    args = parse_args()
    cluster_config = ClusterConfig.model_validate_json(Path(args.config).read_bytes())

    master_addr = cluster_config.nodes[0]
    nnodes = len(cluster_config.nodes)

    for _ in range(args.num_runs):
        processes: list[subprocess.Popen[str]] = []
        for i, node in list(enumerate(cluster_config.nodes))[::-1]:
            torchrun_cmd = " ".join(
                [
                    f"cd {cluster_config.project_dir.as_posix()} &&",
                    cluster_config.torchrun_path.as_posix(),
                    f"--nnodes={nnodes}",
                    f"--node_rank={i}",
                    f"--nproc_per_node={cluster_config.gpus_per_node}",
                    f"--master_addr={master_addr}",
                    f"--master_port={cluster_config.master_port}",
                    args.script,
                    *args.script_args,
                ]
            )
            if i == 0:
                print("Running main command:", torchrun_cmd)
                subprocess.run(torchrun_cmd, shell=True, check=True, executable="/bin/bash")  # noqa: S602
            else:
                ssh_cmd = " ".join(["ssh", node, f"'{torchrun_cmd}'"])
                print("Running worker command:", ssh_cmd)
                processes.append(
                    subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, text=True, shell=True)  # noqa: S602
                )

        print("Main process completed. Waiting for worker completion...")
        for process in processes:
            return_code = process.wait()
            print(f"Worker exited with return code {return_code}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] Aborting experiments due to KeyboardInterrupt.")
        subprocess.run(["/usr/bin/pkill", "-f", "torchrun"], check=True)
        print("[INFO] All torchrun processes have been terminated.")
        raise
