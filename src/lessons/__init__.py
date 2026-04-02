import importlib
import pkgutil
import re
from pathlib import Path

_LESSON_PATTERN = re.compile(r"^s\d+e\d+$")


def _discover_lessons() -> dict:
    lessons = {}
    package_dir = Path(__file__).parent
    for _, name, _ in pkgutil.iter_modules([str(package_dir)]):
        if _LESSON_PATTERN.match(name):
            module = importlib.import_module(f"src.lessons.{name}")
            if hasattr(module, "main"):
                lessons[name] = module.main
    return lessons


lessons = _discover_lessons()


def available_lessons() -> list[str]:
    return set(lessons.keys())


def run_lesson(lesson: str):
    if lesson in available_lessons():
        lessons[lesson]()


__all__ = ["available_lessons", "run_lesson"]
