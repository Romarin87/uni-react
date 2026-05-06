"""Console entry point for joint task training."""

from .tasks.joint import run_joint_entry


def main() -> None:
    run_joint_entry()


if __name__ == "__main__":
    main()
