from __future__ import annotations


MECHANISM_NAMES = {
    "first_price": "First-price sealed-bid auction",
    "second_price": "Second-price sealed-bid auction",
}


def mechanism_description(mechanism: str, treatment: str) -> str:
    name = MECHANISM_NAMES[mechanism]
    if treatment == "name_only":
        return name
    payment = "their own bid" if mechanism == "first_price" else "the second-highest bid"
    if treatment == "concise":
        return f"The highest bidder wins and pays {payment}."
    if treatment == "full":
        return (
            "Bidders receive private values and submit bids simultaneously without seeing "
            f"current-round bids. The highest bidder wins and pays {payment}. The winner's "
            "payoff is private value minus payment; all other bidders receive zero. Ties are "
            "resolved by the configured deterministic rule."
        )
    raise ValueError(f"Unsupported mechanism-description treatment: {treatment}")

