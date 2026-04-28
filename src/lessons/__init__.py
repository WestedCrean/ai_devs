import importlib
import pkgutil
import re
from pathlib import Path

_LESSON_PATTERN = re.compile(r"^s\d+e\d+$")


def _discover_lessons() -> set[str]:
    lessons: set[str] = set()
    package_dir = Path(__file__).parent
    for _, name, _ in pkgutil.iter_modules([str(package_dir)]):
        if _LESSON_PATTERN.match(name):
            lessons.add(name)
    return lessons


lessons = _discover_lessons()


def available_lessons() -> list[str]:
    return sorted(lessons)


def run_lesson(lesson: str):
    if lesson in available_lessons():
        module = importlib.import_module(f"src.lessons.{lesson}")
        if hasattr(module, "main"):
            module.main()


__all__ = ["available_lessons", "run_lesson"]
