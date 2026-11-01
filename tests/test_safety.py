import pytest

from shifa.agents.safety import detect_script, emergency_match

EMERGENCIES = [
    "my father has severe chest pain",
    "Ammi ko seenay mein dard ho raha hai",
    "abbu ko saans nahi aa rahi",
    "bacha behosh ho gaya hai",
    "bachay ko jhatke lag rahe hain",
    "khoon ruk nahi raha",
    "میرے والد کے سینے میں درد ہے",
    "امی کو سانس نہیں آ رہی",
    "my son swallowed some pills, I think it's an overdose",
]

NOT_EMERGENCIES = [
    "Dr Ayesha ki fees kitni hai?",
    "kal subah appointment chahiye",
    "I have a mild cough since two days",
    "bukhar hai halka sa",
    "what are your timings on Saturday",
    "I need a medical fit certificate for my job",
]


@pytest.mark.parametrize("text", EMERGENCIES)
def test_emergency_recall(text):
    assert emergency_match(text)


@pytest.mark.parametrize("text", NOT_EMERGENCIES)
def test_no_false_alarm_on_routine(text):
    assert emergency_match(text) is None


@pytest.mark.parametrize("text,script", [
    ("Dr Ayesha ki fees kitni hai?", "roman_ur"),
    ("What time do you open?", "en"),
    ("ڈاکٹر صاحب کب بیٹھتے ہیں", "ur"),
    ("kal appointment chahiye", "roman_ur"),
    # Real failures from the first manual test session:
    ("mere pass to sirf 500 ruppee hain", "roman_ur"),
    ("mere abbu behosh ho gaye hain", "roman_ur"),
    ("do you accept life insurance as fees", "en"),
])
def test_detect_script(text, script):
    assert detect_script(text) == script


def test_language_is_sticky_on_ambiguous_turns():
    assert detect_script("Rabia Baji", previous="roman_ur") == "roman_ur"
    assert detect_script("5 PM", previous="roman_ur") == "roman_ur"
    assert detect_script("hina baji are you drunk", previous="roman_ur") == "roman_ur"


def test_explicit_language_request_wins():
    # Verbatim from a voice test: English speech that Whisper wrote in Urdu script.
    assert detect_script("میں آپ کو انگلیش میں بات کر رہا ہوں کیوں آپ انگلیش میں نہیں بات کر رہے ہیں؟", "ur") == "en"
    assert detect_script("please urdu mein baat karein", previous="en") == "roman_ur"


def test_language_switches_with_clear_margin():
    assert detect_script("I want to book an appointment for my son please", previous="roman_ur") == "en"
