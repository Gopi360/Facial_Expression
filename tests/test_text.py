import pytest

from app.core.text import face_count_text, person_line
from app.core.types import Box, Person, Reading


def person(label, confidence=92, letter="A"):
    return Person(letter=letter, box=Box(0, 0, 1, 1), reading=Reading(label, confidence), visible=True)


def test_face_count_words_0_1_and_2():
    assert face_count_text(0) == "No faces found"
    assert face_count_text(1) == "1 face found"
    assert face_count_text(2) == "2 faces found"


@pytest.mark.parametrize(
    "label,line",
    [
        ("sleeping", "Person A is sleeping 92%"),
        ("drowsy", "Person A is about to sleep 92%"),
        ("yawning", "Person A is yawning 92%"),
        ("angry", "Person A is angry 92%"),
        ("laughing", "Person A is laughing 92%"),
        ("smiling", "Person A is smiling 92%"),
        ("crying", "Person A is crying 92%"),
        ("neutral", "Person A looks neutral 92%"),
    ],
)
def test_person_line_words_all_labels(label, line):
    assert person_line(person(label)) == line


def test_far_person_line_has_no_percent():
    assert person_line(person("far", 0, letter="B")) == "Person B is too far to read"


def test_person_without_letter_shows_identifying():
    assert person_line(person("neutral", letter=None)) == "Identifying..."
