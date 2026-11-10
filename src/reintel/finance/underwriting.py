"""Rental investment underwriting.

All money is in GBP (£). All rates are decimals (0.05 = 5%). Periods are years
unless the name says otherwise.

Definitions (as used everywhere in the platform)
-----------------------------------------------
* Gross rent            = monthly rent × 12
* Effective rent        = gross rent × (1 − vacancy rate)
* Operating expenses    = management (% of effective rent) + maintenance (% of
                          gross rent) + insurance + ground rent/service charge +
                          other fixed costs. Excludes debt service, income tax
                          and capital expenditure reserves are a separate line.
* NOI                   = effective rent − operating expenses − reserves
* Annual debt service   = 12 × monthly mortgage payment
* Pre-tax cash flow     = NOI − annual debt service
* Gross rental yield    = gross rent ÷ purchase price
* Net yield             = NOI ÷ purchase price
* Cap rate              = NOI ÷ total acquisition cost
                          (purchase price + SDLT + other closing costs + initial
                          refurbishment). Stated explicitly because the
                          denominator differs between firms.
* Cash-on-cash          = pre-tax cash flow ÷ initial cash invested
                          (deposit + SDLT + other closing costs + refurbishment
                          + mortgage fees paid upfront)
* DSCR                  = NOI ÷ annual debt service (∞ for an all-cash purchase)
* IRR                   = annual internal rate of return on the cash invested,
                          including sale proceeds net of selling costs and the
                          outstanding loan at exit. Pre-tax.

These are informational estimates, not investment advice.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

# ---------------------------------------------------------------------------
# Stamp Duty Land Tax (England & Northern Ireland, residential)
# Source: https://www.gov.uk/stamp-duty-land-tax/residential-property-rates
# (rates from 1 April 2025; page checked November 2026). Wales (LTT) and
# Scotland (LBTT) use different taxes and are out of scope.
# ---------------------------------------------------------------------------
SDLT_BANDS = ((125_000, 0.00), (250_000, 0.02), (925_000, 0.05), (1_500_000, 0.10), (float("inf"), 0.12))
SDLT_ADDITIONAL_SURCHARGE = 0.05          # buyer already owns a residential property
SDLT_NON_RESIDENT_SURCHARGE = 0.02
SDLT_SURCHARGE_MIN_PRICE = 40_000          # additional-dwelling rates do not apply below this


def stamp_duty(price: float, additional_property: bool = True, non_resident: bool = False) -> float:
    """SDLT payable on a residential purchase (England / Northern Ireland)."""
    if price < 0:
        raise ValueError("Price cannot be negative")
    surcharge = 0.0
    if additional_property and price >= SDLT_SURCHARGE_MIN_PRICE:
        surcharge += SDLT_ADDITIONAL_SURCHARGE
    if non_resident:
        surcharge += SDLT_NON_RESIDENT_SURCHARGE
    tax, lower = 0.0, 0.0
    for upper, rate in SDLT_BANDS:
        if price > lower:
            tax += (min(price, upper) - lower) * (rate + surcharge)
        lower = upper
    return round(tax, 2)


# ---------------------------------------------------------------------------
# Mortgage maths
# ---------------------------------------------------------------------------
def monthly_mortgage_payment(principal: float, annual_rate: float, years: int,
                             interest_only: bool = False) -> float:
    """Monthly payment. 0% rate repays the principal in equal instalments."""
    if principal < 0 or annual_rate < 0 or years <= 0:
        raise ValueError("Invalid mortgage inputs")
    if principal == 0:
        return 0.0
    r = annual_rate / 12
    if interest_only:
        return principal * r
    n = years * 12
    return principal / n if r == 0 else principal * r / (1 - (1 + r) ** (-n))


def loan_balance(principal: float, annual_rate: float, years: int, months_paid: int,
                 interest_only: bool = False) -> float:
    """Outstanding balance after ``months_paid`` payments."""
    if interest_only or principal == 0:
        return float(principal)
    n = years * 12
    k = min(months_paid, n)
    r = annual_rate / 12
    if r == 0:
        return principal * (1 - k / n)
    pmt = monthly_mortgage_payment(principal, annual_rate, years)
    return principal * (1 + r) ** k - pmt * ((1 + r) ** k - 1) / r


def irr(cash_flows: list[float] | np.ndarray, lo: float = -0.99, hi: float = 10.0, tol: float = 1e-10) -> float:
    """Internal rate of return by bisection (robust; returns nan if no sign change)."""
    cf = np.asarray(cash_flows, dtype=float)

    def npv(r):
        return float(np.sum(cf / (1 + r) ** np.arange(len(cf))))

    f_lo, f_hi = npv(lo), npv(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return float("nan")
    for _ in range(300):
        mid = (lo + hi) / 2
        f_mid = npv(mid)
        if abs(f_mid) < tol:
            return mid
        if f_lo * f_mid < 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Deal model
# ---------------------------------------------------------------------------
@dataclass
class Assumptions:
    # purchase
    purchase_price: float = 200_000.0
    additional_property: bool = True          # SDLT 5% surcharge (typical for a buy-to-let investor)
    non_resident: bool = False
    legal_and_survey: float = 2_500.0
    refurbishment: float = 0.0
    # income
    monthly_rent: float = 1_000.0
    vacancy_rate: float = 0.05
    # operating costs (annual unless stated)
    management_pct: float = 0.10              # of effective rent
    maintenance_pct: float = 0.05             # of gross rent
    insurance: float = 400.0
    service_charge_ground_rent: float = 0.0
    other_costs: float = 200.0
    reserves_pct: float = 0.02                # capex reserve, % of gross rent
    # financing
    loan_to_value: float = 0.75               # 0 = all cash
    mortgage_rate: float = 0.055
    mortgage_years: int = 25
    interest_only: bool = True                # most UK buy-to-let mortgages
    mortgage_fees: float = 1_500.0            # arrangement fee paid upfront
    # growth and exit
    hold_years: int = 5
    rent_growth: float = 0.03
    cost_inflation: float = 0.03
    price_growth: float = 0.02
    selling_costs_pct: float = 0.02           # agent + legal on exit

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Result:
    sdlt: float
    acquisition_cost: float
    loan: float
    cash_invested: float
    gross_rent: float
    effective_rent: float
    operating_expenses: float
    reserves: float
    noi: float
    monthly_payment: float
    debt_service: float
    cash_flow: float
    gross_yield: float
    net_yield: float
    cap_rate: float
    cash_on_cash: float
    dscr: float
    break_even_occupancy: float
    exit_value: float
    exit_loan_balance: float
    exit_equity: float
    irr: float
    equity_multiple: float
    yearly: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _year_one(a: Assumptions, rent_factor: float = 1.0, cost_factor: float = 1.0):
    gross = a.monthly_rent * 12 * rent_factor
    effective = gross * (1 - a.vacancy_rate)
    opex = (a.management_pct * effective + a.maintenance_pct * gross
            + (a.insurance + a.service_charge_ground_rent + a.other_costs) * cost_factor)
    reserves = a.reserves_pct * gross
    return gross, effective, opex, reserves


def analyse(a: Assumptions) -> Result:
    """Underwrite a buy-to-let purchase. Validates inputs and returns every metric."""
    if a.purchase_price <= 0:
        raise ValueError("Purchase price must be positive")
    if not 0 <= a.vacancy_rate <= 1:
        raise ValueError("Vacancy rate must be between 0 and 1")
    if not 0 <= a.loan_to_value < 1:
        raise ValueError("Loan-to-value must be at least 0 and below 1")
    if a.monthly_rent < 0 or a.mortgage_rate < 0 or a.hold_years < 1:
        raise ValueError("Invalid rent, rate or hold period")

    sdlt = stamp_duty(a.purchase_price, a.additional_property, a.non_resident)
    acquisition = a.purchase_price + sdlt + a.legal_and_survey + a.refurbishment
    loan = a.purchase_price * a.loan_to_value
    fees = a.mortgage_fees if loan > 0 else 0.0
    cash_in = acquisition - loan + fees
    pmt = monthly_mortgage_payment(loan, a.mortgage_rate, a.mortgage_years, a.interest_only)
    debt = 12 * pmt

    gross, effective, opex, reserves = _year_one(a)
    noi = effective - opex - reserves
    cash_flow = noi - debt

    # Break-even occupancy: share of gross rent needed to cover all costs and debt.
    # Management is a % of collected rent, so solve: o·G(1−m) − maint − fixed − res − debt = 0
    fixed = a.maintenance_pct * gross + a.insurance + a.service_charge_ground_rent + a.other_costs + reserves
    denom = gross * (1 - a.management_pct)
    be = (fixed + debt) / denom if denom > 0 else float("inf")

    # Multi-year projection
    yearly, flows = [], [-cash_in]
    for y in range(1, a.hold_years + 1):
        rf = (1 + a.rent_growth) ** (y - 1)
        cf_ = (1 + a.cost_inflation) ** (y - 1)
        g, e, o, r = _year_one(a, rf, cf_)
        n = e - o - r
        c = n - debt
        row = {"year": y, "gross_rent": g, "effective_rent": e, "operating_expenses": o,
               "reserves": r, "noi": n, "debt_service": debt, "cash_flow": c}
        if y == a.hold_years:
            value = a.purchase_price * (1 + a.price_growth) ** a.hold_years
            bal = loan_balance(loan, a.mortgage_rate, a.mortgage_years, 12 * a.hold_years, a.interest_only)
            proceeds = value * (1 - a.selling_costs_pct) - bal
            row.update({"sale_value": value, "selling_costs": value * a.selling_costs_pct,
                        "loan_repaid": bal, "sale_proceeds": proceeds})
            flows.append(c + proceeds)
        else:
            flows.append(c)
        yearly.append(row)

    exit_value = yearly[-1]["sale_value"]
    exit_bal = yearly[-1]["loan_repaid"]
    exit_equity = yearly[-1]["sale_proceeds"]
    total_back = sum(flows[1:])
    return Result(
        sdlt=sdlt, acquisition_cost=acquisition, loan=loan, cash_invested=cash_in,
        gross_rent=gross, effective_rent=effective, operating_expenses=opex, reserves=reserves,
        noi=noi, monthly_payment=pmt, debt_service=debt, cash_flow=cash_flow,
        gross_yield=gross / a.purchase_price, net_yield=noi / a.purchase_price,
        cap_rate=noi / acquisition,
        cash_on_cash=cash_flow / cash_in if cash_in > 0 else float("nan"),
        dscr=noi / debt if debt > 0 else float("inf"),
        break_even_occupancy=be,
        exit_value=exit_value, exit_loan_balance=exit_bal, exit_equity=exit_equity,
        irr=irr(flows), equity_multiple=total_back / cash_in if cash_in > 0 else float("nan"),
        yearly=yearly,
    )


# ---------------------------------------------------------------------------
# Scenario presets
# ---------------------------------------------------------------------------
SCENARIO_DELTAS = {
    "Base": {},
    "Upside": {"vacancy_rate": -0.02, "mortgage_rate": -0.01, "rent_growth": +0.015,
               "price_growth": +0.02, "cost_inflation": -0.005},
    "Downside": {"vacancy_rate": +0.07, "mortgage_rate": +0.02, "rent_growth": -0.03,
                 "price_growth": -0.04, "cost_inflation": +0.02, "selling_costs_pct": +0.01},
}


def apply_scenario(base: Assumptions, deltas: dict) -> Assumptions:
    changes = {}
    for k, d in deltas.items():
        v = getattr(base, k) + d
        if k in ("vacancy_rate", "mortgage_rate", "selling_costs_pct"):
            v = max(0.0, v)
        if k == "vacancy_rate":
            v = min(1.0, v)
        changes[k] = v
    return replace(base, **changes)


def scenarios(base: Assumptions, deltas: dict[str, dict] | None = None) -> dict[str, Result]:
    deltas = deltas or SCENARIO_DELTAS
    return {name: analyse(apply_scenario(base, d)) for name, d in deltas.items()}


SENSITIVITY = {
    "monthly_rent": ("Rent ±10%", "rel", 0.10),
    "vacancy_rate": ("Vacancy ±5 pts", "abs", 0.05),
    "mortgage_rate": ("Mortgage rate ±1.5 pts", "abs", 0.015),
    "purchase_price": ("Purchase price ±10%", "rel", 0.10),
    "price_growth": ("House price growth ±3 pts/yr", "abs", 0.03),
    "cost_inflation": ("Cost inflation ±2 pts/yr", "abs", 0.02),
}


def sensitivity(base: Assumptions, metric: str = "irr") -> list[dict]:
    """One-at-a-time sensitivity of a metric (tornado chart data)."""
    base_v = getattr(analyse(base), metric)
    out = []
    for k, (label, kind, size) in SENSITIVITY.items():
        vals = []
        for sign in (-1, 1):
            v = getattr(base, k)
            v = v * (1 + sign * size) if kind == "rel" else v + sign * size
            if k in ("vacancy_rate", "mortgage_rate"):
                v = min(max(v, 0.0), 1.0)
            vals.append(getattr(analyse(replace(base, **{k: v})), metric))
        out.append({"driver": label, "low": vals[0], "high": vals[1], "base": base_v,
                    "swing": abs(vals[1] - vals[0])})
    return sorted(out, key=lambda r: r["swing"], reverse=True)


# ---------------------------------------------------------------------------
# Vectorised year-one screen (Opportunity Finder ranks thousands of properties)
# ---------------------------------------------------------------------------
def screen(prices, monthly_rents, a: Assumptions):
    """Year-one metrics for many properties at once, using the same formulas as ``analyse``.

    Returns a dict of numpy arrays. A test checks it against ``analyse`` row by row.
    """
    price = np.asarray(prices, dtype=float)
    rent = np.asarray(monthly_rents, dtype=float)
    sd = np.array([stamp_duty(p, a.additional_property, a.non_resident) for p in price])
    acq = price + sd + a.legal_and_survey + a.refurbishment
    loan = price * a.loan_to_value
    fees = np.where(loan > 0, a.mortgage_fees, 0.0)
    cash_in = acq - loan + fees
    if a.interest_only:
        pmt = loan * a.mortgage_rate / 12
    elif a.mortgage_rate == 0:
        pmt = loan / (a.mortgage_years * 12)
    else:
        r = a.mortgage_rate / 12
        pmt = loan * r / (1 - (1 + r) ** (-a.mortgage_years * 12))
    debt = 12 * pmt
    gross = rent * 12
    eff = gross * (1 - a.vacancy_rate)
    opex = (a.management_pct * eff + a.maintenance_pct * gross
            + a.insurance + a.service_charge_ground_rent + a.other_costs)
    res = a.reserves_pct * gross
    noi = eff - opex - res
    cf = noi - debt
    with np.errstate(divide="ignore", invalid="ignore"):
        return {
            "sdlt": sd, "acquisition_cost": acq, "cash_invested": cash_in, "gross_yield": gross / price,
            "net_yield": noi / price, "noi": noi, "debt_service": debt, "cash_flow": cf,
            "cap_rate": noi / acq, "cash_on_cash": cf / cash_in,
            "dscr": np.where(debt > 0, noi / np.where(debt > 0, debt, 1), np.inf),
        }
