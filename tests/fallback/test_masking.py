"""RegexMasker and the Masker protocol."""

import pytest

from slmstrict.fallback import Masker, RegexMasker


def test_regex_masker_satisfies_the_protocol() -> None:
    assert isinstance(RegexMasker(), Masker)


def test_masks_email_addresses() -> None:
    out = RegexMasker().mask("Mail ayse.yilmaz+x@example.com.tr or bob@corp.io today")
    assert out == "Mail [EMAIL] or [EMAIL] today"


def test_masks_ibans_of_known_countries_with_and_without_spaces() -> None:
    masker = RegexMasker()
    assert masker.mask("IBAN TR33 0006 1005 1978 6457 8413 26 ok") == "IBAN [IBAN] ok"
    assert masker.mask("iban: tr330006100519786457841326.") == "iban: [IBAN]."
    assert masker.mask("DE89 3704 0044 0532 0130 00") == "[IBAN]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        # Length is a multiple of 4: the greedy match used to swallow the next word and then fail
        # the checksum, leaving the whole IBAN in the clear.
        ("IBAN BE68 5390 0754 7034 Euro", "IBAN [IBAN] Euro"),
        ("IBAN BE68539007547034 ok", "IBAN [IBAN] ok"),
        ("BE68 5390 0754 7034 and AT61 1904 3002 3457 3201 next", "[IBAN] and [IBAN] next"),
        ("GB82 WEST 1234 5698 7654 32 thanks", "[IBAN] thanks"),
        # A failed checksum (typo) is still an account number.
        ("IBAN TR33 0006 1005 1978 6457 8413 27 ok", "IBAN [IBAN] ok"),
        # Truncated.
        ("IBAN TR33 0006 1005 1978", "IBAN [IBAN]"),
    ],
)
def test_iban_is_masked_when_the_neighbourhood_or_checksum_is_odd(text: str, expected: str) -> None:
    assert RegexMasker().mask(text) == expected


def test_an_iban_lookalike_with_too_few_characters_is_left_alone() -> None:
    assert RegexMasker().mask("room TR33 and AT61 are free") == "room TR33 and AT61 are free"


def test_masks_a_card_number() -> None:
    masker = RegexMasker()
    assert masker.mask("card 4111 1111 1111 1111 expires") == "card [CARD] expires"
    assert masker.mask("card 4111-1111-1111-1111") == "card [CARD]"
    assert masker.mask("card 4111111111111111.") == "card [CARD]."
    assert masker.mask("amex 3782 822463 10005 ok") == "amex [CARD] ok"


@pytest.mark.parametrize(
    "text",
    [
        "card 4111 1111 1111 1111 123",  # CVV glued to the number
        "card 4111 1111 1111 1111 123 expires 12/29",
        "card 4111-1111-1111-1111-123",
        "card 4111111111111111123",
        "card 4111 1111 1111 1111 1234",
        "card 5555 5555 5555 4444 999",
        "card 1234 5678 9012 3456",  # fails Luhn: still masked
        "card 1234567890123",  # 13 digits, no checksum
    ],
)
def test_card_numbers_are_masked_with_a_cvv_or_a_bad_checksum(text: str) -> None:
    out = RegexMasker().mask(text)
    assert out == "card [CARD]" or out.startswith("card [CARD] ")
    assert not any(ch.isdigit() for ch in out.split("expires")[0])


def test_masks_tckn_with_or_without_a_valid_checksum() -> None:
    masker = RegexMasker()
    assert masker.mask("TC 10000000146 verified") == "TC [TCKN] verified"
    assert masker.mask("TC 12345678901 verified") == "TC [TCKN] verified"
    assert masker.mask("ref 10000000147") == "ref [TCKN]"


def test_dates_and_short_numbers_are_not_cards() -> None:
    text = "from 2026-11-16 2026-11-17 qty 1234 5678 9012"
    assert RegexMasker().mask(text) == text


def test_masks_phone_numbers() -> None:
    masker = RegexMasker()
    assert masker.mask("call 05321234567 now") == "call [PHONE] now"
    assert masker.mask("call +905321234567 now") == "call [PHONE] now"
    assert masker.mask("call 0(532) 123 45 67 now") == "call [PHONE] now"
    assert masker.mask("call 0532 123 45 67 now") == "call [PHONE] now"
    assert masker.mask("call +90 532 123 45 67 now") == "call [PHONE] now"
    assert masker.mask("call +49 170 1234567 now") == "call [PHONE] now"


def test_plain_text_and_small_numbers_are_untouched() -> None:
    text = "Order 42 shipped on 2026-11-14 to the warehouse, 3 items."
    assert RegexMasker().mask(text) == text


def test_kinds_restricts_the_recognizers() -> None:
    masker = RegexMasker(kinds=["email"])
    assert masker.mask("a@b.co 0532 123 45 67") == "[EMAIL] 0532 123 45 67"


def test_unknown_kind_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown masking kinds: passport"):
        RegexMasker(kinds=["email", "passport"])


def test_masking_is_idempotent() -> None:
    masker = RegexMasker()
    once = masker.mask("x@y.com and 4111 1111 1111 1111")
    assert masker.mask(once) == once


@pytest.mark.parametrize(
    "text",
    [
        "card 4111\t1111\t1111\t1111",  # TSV / pasted table row
        "card 4111  1111 1111 1111",  # double space (OCR, form output)
        "card 4111_1111_1111_1111",
        "card 4111/1111/1111/1111",
        "card 4111 - 1111 - 1111 - 1111",
        "card 4111\u00a01111\u00a01111\u00a01111",
        "card 4111.1111.1111.1111",
        "card 4111\n1111\n1111\n1111",
        "card 4111111111111111 123",  # contiguous number + CVV
        "card 4111  1111  1111  1111  123",
    ],
)
def test_card_separators_do_not_decide_whether_it_is_masked(text: str) -> None:
    assert RegexMasker().mask(text) == "card [CARD]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IBAN NL91-ABNA-0417-1643-00", "IBAN [IBAN]"),
        ("IBAN NL91  ABNA 0417 1643 00", "IBAN [IBAN]"),
        ("IBAN NL91 ABNA 0417 1643 00 ok", "IBAN [IBAN] ok"),
        ("IBAN NL91\tABNA\t0417\t1643\t00.", "IBAN [IBAN]."),
        ("IBAN NL91_ABNA_0417_1643_00", "IBAN [IBAN]"),
        ("IBAN NL91ABNA0417164300", "IBAN [IBAN]"),
        ("IBAN TR33\u00a00006\u00a01005\u00a01978\u00a06457\u00a08413\u00a026", "IBAN [IBAN]"),
        ("DE89/3704/0044/0532/0130/00 x", "[IBAN] x"),
        ("IBAN GB82-WEST-1234-5698-7654-32 thanks", "IBAN [IBAN] thanks"),
        ("IBAN BE68-5390-0754-7034 Euro", "IBAN [IBAN] Euro"),
    ],
)
def test_iban_separators_do_not_decide_whether_it_is_masked(text: str, expected: str) -> None:
    assert RegexMasker().mask(text) == expected


def test_contiguous_and_spaced_card_with_cvv_are_masked_alike() -> None:
    masker = RegexMasker()
    assert masker.mask("card 4111111111111111 123") == masker.mask("card 4111 1111 1111 1111 123")


@pytest.mark.parametrize(
    "text",
    [
        "AT12 pieces were sold",
        "see TR33 and AT61 are free",
        "NL91 is a code for the shop",
    ],
)
def test_ordinary_words_after_a_country_code_are_not_an_iban(text: str) -> None:
    assert RegexMasker().mask(text) == text


def test_phone_separators_are_generic_too() -> None:
    assert RegexMasker().mask("call 0532\t123\t45\t67 now") == "call [PHONE] now"
    assert RegexMasker().mask("call 0532/123/45/67 now") == "call [PHONE] now"


def test_the_docstring_says_this_is_a_best_effort_example() -> None:
    doc = RegexMasker.__doc__ or ""
    assert "best-effort example implementation" in doc
    assert "McpGuard-compatible" in doc


def test_long_hostile_inputs_are_masked_in_reasonable_time() -> None:
    import time

    masker = RegexMasker()
    started = time.perf_counter()
    for text in (
        "9" * 100_000,
        "a" * 100_000,
        "1234 " * 20_000,
        "- " * 50_000,
        "XX12 " * 20_000,  # IBAN-shaped, unknown country
        "1234" + " " * 8 + "5678 " * 20_000,
    ):
        masker.mask(text)
    assert time.perf_counter() - started < 5


@pytest.mark.parametrize(
    "text",
    [
        "card 4111    1111 1111 1111",  # 4 spaces: fixed-width table, OCR
        "card 4111        1111  1111 1111",  # 8 spaces
        "card 4111 \t | 1111 | 1111 | 1111",
        "card 4111\n\n1111\n\n1111\n\n1111",
    ],
)
def test_wide_gaps_up_to_the_separator_limit_do_not_hide_a_card(text: str) -> None:
    assert RegexMasker().mask(text) == "card [CARD]"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IBAN NL91    ABNA 0417 1643 00", "IBAN [IBAN]"),
        ("IBAN NL91        ABNA 0417 1643 00 ok", "IBAN [IBAN] ok"),
        ("IBAN TR33     0006     1005 1978 6457 8413 26", "IBAN [IBAN]"),
    ],
)
def test_wide_gaps_up_to_the_separator_limit_do_not_hide_an_iban(text: str, expected: str) -> None:
    assert RegexMasker().mask(text) == expected


def test_the_docstring_states_the_separator_limit() -> None:
    doc = RegexMasker.__doc__ or ""
    assert "1-8 characters" in doc
    assert "longer gap" in doc
    assert "is not recognized" in doc


@pytest.mark.parametrize(
    "text",
    [
        "years 2020, 2021, 2022, 2023",
        "years 2019; 2020; 2021; 2022",
        "sizes 1000, 1500, 2000, 2500 sold",
        "years 2018, 2019, 2020, 2021",
    ],
)
def test_comma_lists_of_numbers_that_are_not_cards_are_left_alone(text: str) -> None:
    assert RegexMasker().mask(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "card 4111, 1111, 1111, 1111",
        "card 4111, 1111, 1111, 1111, 123",
        "card 4111; 1111; 1111; 1111",
    ],
)
def test_a_card_written_with_commas_is_still_masked(text: str) -> None:
    assert RegexMasker().mask(text) == "card [CARD]"


def test_hyphen_and_dot_groupings_stay_masked_even_without_a_valid_checksum() -> None:
    masker = RegexMasker()
    assert masker.mask("range 1000-2000-3000-4000") == "range [CARD]"
    assert masker.mask("ip 1921.1681.1001.2001") == "ip [CARD]"


def _iban_with_check_digits(country: str, account: str) -> str:
    rearranged = account + "".join(str(int(ch, 36)) for ch in country) + "00"
    return f"{country}{98 - int(rearranged) % 97:02d}{account}"


def test_unknown_country_iban_is_masked_only_with_a_valid_checksum() -> None:
    masker = RegexMasker()
    valid = _iban_with_check_digits("ZZ", "1234567890123456")
    assert masker.mask(f"IBAN {valid} ok") == "IBAN [IBAN] ok"
    spaced = " ".join(valid[i : i + 4] for i in range(0, len(valid), 4))
    assert masker.mask(f"IBAN {spaced} ok") == "IBAN [IBAN] ok"
    broken = valid[:-1] + ("0" if valid[-1] != "0" else "1")
    # Without a valid checksum the IBAN rule does not fire (the digits still fall to the card rule).
    assert "[IBAN]" not in masker.mask(f"IBAN {broken} ok")
