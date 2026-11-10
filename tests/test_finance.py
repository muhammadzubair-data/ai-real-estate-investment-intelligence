"""Worked-example tests for the underwriting engine.

Every expected number below is worked out by hand in the comments, so a reviewer
can check the formulas without running the code.
"""
import math

import pytest

from reintel.finance.underwriting import (
    Assumptions,
    analyse,
    apply_scenario,
    irr,
    loan_balance,
    monthly_mortgage_payment,
    scenarios,
    sensitivity,
    stamp_duty,
)


# --- Stamp duty (England / NI residential) ---------------------------------------------
def test_sdlt_standard_rates():
    # £300,000, main home: 0% on first 125k, 2% on next 125k (2,500), 5% on last 50k (2,500)
    assert stamp_duty(300_000, additional_property=False) == 5_000
    # £500,000: 0 + 2,500 + 5% × 250k (12,500)
    assert stamp_duty(500_000, additional_property=False) == 15_000
    assert stamp_duty(125_000, additional_property=False) == 0


def test_sdlt_additional_property_surcharge():
    # £300,000 with +5%: 5% × 125k (6,250) + 7% × 125k (8,750) + 10% × 50k (5,000)
    assert stamp_duty(300_000, additional_property=True) == 20_000
    # below £40,000 the higher rates do not apply
    assert stamp_duty(39_000, additional_property=True) == 0
    # non-resident adds a further 2%: 300k × 2% = 6,000 on top
    assert stamp_duty(300_000, additional_property=True, non_resident=True) == 26_000


# --- Mortgage ---------------------------------------------------------------------------
def test_zero_interest_mortgage():
    assert monthly_mortgage_payment(12_000, 0, 1) == 1_000
    assert loan_balance(12_000, 0, 1, 6) == pytest.approx(6_000)


def test_repayment_mortgage_matches_annuity_formula():
    # £100,000 at 6% over 25 years: 100000 × 0.005 / (1 − 1.005^−300) = £644.30
    assert monthly_mortgage_payment(100_000, 0.06, 25) == pytest.approx(644.30, abs=0.01)
    # fully repaid at term
    assert loan_balance(100_000, 0.06, 25, 300) == pytest.approx(0, abs=1e-6)


def test_interest_only_mortgage():
    # £150,000 at 5.5% interest-only: 150000 × 0.055 / 12 = £687.50, balance unchanged
    assert monthly_mortgage_payment(150_000, 0.055, 25, interest_only=True) == pytest.approx(687.50)
    assert loan_balance(150_000, 0.055, 25, 60, interest_only=True) == 150_000


def test_invalid_mortgage_inputs():
    with pytest.raises(ValueError):
        monthly_mortgage_payment(-1, 0.05, 25)
    with pytest.raises(ValueError):
        monthly_mortgage_payment(100_000, 0.05, 0)


# --- Full deal: all-cash worked example ----------------------------------------------------
BASE = dict(purchase_price=100_000, additional_property=True, legal_and_survey=2_000, refurbishment=0,
            monthly_rent=800, vacancy_rate=0.05, management_pct=0.10, maintenance_pct=0.05,
            insurance=300, service_charge_ground_rent=0, other_costs=200, reserves_pct=0.02,
            mortgage_fees=1_000, hold_years=1, rent_growth=0, cost_inflation=0, price_growth=0,
            selling_costs_pct=0.02)


def test_all_cash_purchase():
    r = analyse(Assumptions(**BASE, loan_to_value=0))
    # SDLT: £100k is all in the 0% band, +5% surcharge = 5,000
    assert r.sdlt == 5_000
    # acquisition = 100,000 + 5,000 + 2,000
    assert r.acquisition_cost == 107_000
    # gross 800×12 = 9,600; effective 9,600 × 0.95 = 9,120
    assert r.gross_rent == 9_600 and r.effective_rent == pytest.approx(9_120)
    # opex: mgmt 912 + maintenance 480 + fixed 500 = 1,892; reserves 192
    assert r.operating_expenses == pytest.approx(1_892)
    assert r.reserves == pytest.approx(192)
    # NOI = 9,120 − 1,892 − 192 = 7,036
    assert r.noi == pytest.approx(7_036)
    assert r.loan == 0 and r.debt_service == 0 and math.isinf(r.dscr)
    assert r.cash_invested == 107_000                    # no mortgage fee without a mortgage
    assert r.gross_yield == pytest.approx(0.096)
    assert r.cap_rate == pytest.approx(7_036 / 107_000)
    assert r.cash_on_cash == pytest.approx(7_036 / 107_000)


def test_financed_purchase():
    r = analyse(Assumptions(**BASE, loan_to_value=0.75, mortgage_rate=0.05, interest_only=True))
    # loan 75,000 interest-only at 5% = 312.50/month, 3,750/yr
    assert r.loan == 75_000 and r.monthly_payment == pytest.approx(312.5)
    # cash in = 107,000 − 75,000 + 1,000 fee = 33,000
    assert r.cash_invested == 33_000
    # cash flow = 7,036 − 3,750 = 3,286; CoC = 3,286 / 33,000
    assert r.cash_flow == pytest.approx(3_286)
    assert r.cash_on_cash == pytest.approx(3_286 / 33_000)
    assert r.dscr == pytest.approx(7_036 / 3_750)
    # one-year hold, no growth: sale 100,000 − 2% = 98,000, repay 75,000 → 23,000
    # total back 3,286 + 23,000 = 26,286 → IRR = 26,286 / 33,000 − 1
    assert r.exit_equity == pytest.approx(23_000)
    assert r.irr == pytest.approx(26_286 / 33_000 - 1, abs=1e-6)
    assert r.equity_multiple == pytest.approx(26_286 / 33_000)


def test_financed_zero_rate_mortgage():
    r = analyse(Assumptions(**BASE, loan_to_value=0.6, mortgage_rate=0.0, interest_only=False,
                            mortgage_years=10))
    # 60,000 over 120 months at 0% = 500/month
    assert r.monthly_payment == pytest.approx(500)
    assert r.debt_service == pytest.approx(6_000)
    # after 12 payments 54,000 remains
    assert r.exit_loan_balance == pytest.approx(54_000)


def test_break_even_occupancy_balances_cash_flow():
    a = Assumptions(**BASE, loan_to_value=0.75, mortgage_rate=0.05)
    r = analyse(a)
    from dataclasses import replace
    r2 = analyse(replace(a, vacancy_rate=1 - r.break_even_occupancy))
    assert r2.cash_flow == pytest.approx(0, abs=1e-6)


def test_input_validation():
    with pytest.raises(ValueError):
        analyse(Assumptions(vacancy_rate=1.5))
    with pytest.raises(ValueError):
        analyse(Assumptions(loan_to_value=1.0))
    with pytest.raises(ValueError):
        analyse(Assumptions(purchase_price=0))


def test_irr_known_values():
    assert irr([-100, 110]) == pytest.approx(0.10)
    assert irr([-1000, 0, 0, 1331]) == pytest.approx(0.10)
    assert math.isnan(irr([100, 100]))


def test_scenarios_are_ordered_and_bounded():
    s = scenarios(Assumptions(hold_years=5))
    assert s["Downside"].irr < s["Base"].irr < s["Upside"].irr
    down = apply_scenario(Assumptions(vacancy_rate=0.0, mortgage_rate=0.0), {"vacancy_rate": -0.1,
                                                                          "mortgage_rate": -0.5})
    assert down.vacancy_rate == 0 and down.mortgage_rate == 0


def test_sensitivity_sorted_by_swing():
    rows = sensitivity(Assumptions())
    swings = [r["swing"] for r in rows]
    assert swings == sorted(swings, reverse=True) and len(rows) == 6


def test_vectorised_screen_matches_full_model():
    from dataclasses import replace

    from reintel.finance.underwriting import screen
    prices = [90_000, 210_000, 480_000]
    rents = [700, 1_100, 2_100]
    for io in (True, False):
        a = Assumptions(loan_to_value=0.7, mortgage_rate=0.052, interest_only=io)
        s = screen(prices, rents, a)
        for i, (p, r) in enumerate(zip(prices, rents)):
            full = analyse(replace(a, purchase_price=p, monthly_rent=r))
            for k in ("sdlt", "noi", "cash_flow", "cap_rate", "cash_on_cash", "dscr", "gross_yield"):
                assert s[k][i] == pytest.approx(getattr(full, k)), k
