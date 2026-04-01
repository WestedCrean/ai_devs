import typer

from src.lessons import available_lessons, run_lesson


def main(lesson_str: str):
    if lesson_str not in available_lessons():
        raise KeyError(f"lesson_str must be one of {available_lessons()}")

    run_lesson(lesson_str)


if __name__ == "__main__":
    typer.run(main)
