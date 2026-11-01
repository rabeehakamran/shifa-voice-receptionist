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


def test_exact_glued_duplicate_removed():
    glued = "براہِ کرم اپنا نام بتائیں۔براہِ کرم اپنا نام بتائیں۔"
    assert clean(glued) == "براہِ کرم اپنا نام بتائیں۔"


def test_paraphrased_repeats_cut_by_sentence_cap():
    # Verbatim from a voice test: three paraphrases of one request, glued without spaces.
    real = ("براہ کرم واضح کریں کہ آپ ڈاکٹر عائشہ ملک کے ساتھ اپائنٹمنٹ چاہتے ہیں۔"
            "آپ کی اپائنٹمنٹ کے لیے ڈاکٹر عائشہ ملک میں سے کسی ایک کا انتخاب کریں۔"
            "آپ کی سہولت کے مطابق ڈاکٹر کا نام اور وقت بتائیں تاکہ میں اپائنٹمنٹ بُک کر سکوں۔")
    assert clean(real).count("۔") == 2


def test_reply_capped_at_two_sentences():
    assert clean("One. Two. Three. Four.") == "One. Two."


def test_abbreviations_and_times_do_not_split_sentences():
    text = "Dr. Ayesha Malik ki fee Rs. 2000 hai. Kal subah 9.20 baje aa jayein."
    assert clean(text) == text


def test_internal_ids_flagged():
    assert problems("آپ کی اپائنٹمنٹ (آئی ڈی 2) منسوخ کر دی گئی ہے۔", "ur")
    assert problems("Booked, appointment_id=4.", "en")


def test_english_turn_rejects_roman_urdu_reply():
    # Verbatim from a voice call with an English-speaking caller.
    assert "must be written in English" in problems("Kal subah 9 baje, 9:20 baje, ya 9:40 baje mein se koi pasand "
                                                    "karenge?", "en")
    assert problems("Your appointment with Dr. Ayesha Malik is confirmed for Saturday 9:40 AM.", "en") == []


def test_normal_reply_passes():
    assert problems("Dr. Ayesha Malik ki fee Rs. 2000 hai.", "roman_ur") == []
