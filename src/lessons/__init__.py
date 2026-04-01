from src.lessons.s01e01 import main as s01e01_main

lessons = {
    "s01e01": s01e01_main,
}


def available_lessons() -> list[str]:
    return set(lessons.keys())


def run_lesson(lesson: str):
    if lesson in available_lessons():
        main_fun = lessons.get(lesson)
        main_fun()


__all__ = ["available_lessons", "run_lesson"]
