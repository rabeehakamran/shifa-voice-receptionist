from shifa.agents.guardrails import clean, problems

# Verbatim shape of the leak seen in the first Arena run (shortened).
LEAK = ("Appointment ​​  \n\n\nThe response must be after ………………\n\n… ……\n\nWe need to respond properly: "
        "confirm appointment details in 2 short sentences, Roman Urdu.Aapka appointment Dr. Ayesha Malik ke saath "
        "kal subah 9 baje confirm ho gaya hai.")


def test_reasoning_leak_is_flagged():
    assert "contains internal reasoning or meta-commentary" in problems(clean(LEAK), "roman_ur")


def test_artifacts_are_cleaned_not_flagged():
    text = clean("Ji, aapka naam kya hai??> (Waiting for name)")
    assert text == "Ji, aapka naam kya hai?"
    assert problems(text, "roman_ur") == []


def test_gendered_honorifics_flagged():
    assert problems("Bhai sahab, aapka naam kya hai?", "roman_ur")
    assert problems("بھائی صاحب، آپ کا نام کیا ہے؟", "ur")
    assert problems("Kaun sa time pasand karenge?", "roman_ur") == []


def test_feminine_verbs_about_a_doctor_are_fine():
    assert problems("Dr. Ayesha aap ko subah dekh sakti hain.", "roman_ur") == []
    assert problems("ڈاکٹر ثنا ہفتے کو بیٹھتی ہیں۔", "ur") == []


def test_urdu_caller_needs_urdu_script():
    assert problems("Ji, kal subah 10 baje slot hai.", "ur") == ["must be written in Urdu script"]
    assert problems("جی، کل صبح دس بجے وقت ہے۔", "ur") == []


def test_normal_reply_passes():
    assert problems("Dr. Ayesha Malik ki fee Rs. 2000 hai.", "roman_ur") == []
