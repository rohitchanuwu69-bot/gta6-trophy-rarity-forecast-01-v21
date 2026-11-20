#!/usr/bin/env python3
"""Trophy Dilution Forecaster v21.

Forecasts, for every trophy of Grand Theft Auto VI on PlayStation 5, its
earn rate on Exophase at the case target, as five quantiles per trophy.

Two regimes, picked per case from the history the house recorded:

* Flat history (every read identical): Exophase has not refreshed yet, so the
  forecast models the next refresh from the one refresh the rules document:
  the page went from 638 to 3868 players tracked, every rate but the 100%
  one lower. The fitted refresh keeps each earned count, adds two players
  for the first 29 trophies in list order, and adds a rate-proportional
  share of the newcomers for the trophies with the top three distinct
  sub-100 rates (20% newcomer earn at the median). All counts are whole
  players, rounded half-up, and every rate is printed to 2 decimals.

* Moving history: a step-aware log-time fit. The largest single-read jump
  in players tracked (over 1.5x) or in any rate (over 0.5 logit) marks a
  refresh; trends are fitted only on the reads after it. When the last
  three reads of the segment are flat, the median is their level instead
  of an extrapolation. Otherwise the median comes from least-squares
  fits of log players and log earned counts against log time since the
  game's release midnight, read at the target.

The spread is honest about what is unknown: a small floor for the flat
regime, and for the moving regime the floor plus a horizon term plus the
fit residual, capped near 1.1 logit in total.
"""

import glob
import json
import math
import os
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP

Z_05 = -1.6448536269514722
Z_25 = -0.6744897501960817
Z_75 = 0.6744897501960817
Z_95 = 1.6448536269514722

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]

# The one refresh the rules document: 638 -> 3868 players tracked.
REFRESH_FROM = 638.0
REFRESH_TO = 3868.0
PREFIX_COUNT = 29
NEWCOMER_MID = 0.20
NEWCOMER_LO = 0.10
NEWCOMER_HI = 0.30

FLAT_SPREAD = 0.0075
FLAT_SPREAD_MIN = 0.0075
FLAT_SPREAD_MAX = 0.0375

MOVE_FLOOR = 0.015
MOVE_HORIZON_RATE = 0.02
MOVE_SPREAD_CAP = 1.1

JUMP_PLAYERS = 1.5
JUMP_RATE = 0.5
FLAT_LOGIT_RANGE = 0.01


def half_up_int(value):
    return int(math.floor(value + 0.5))


def half_up_2dp(value):
    cents = Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(cents)


def clamp_rate(rate):
    if rate is None:
        return None
    if rate < 0.01:
        return 0.01
    if rate > 99.99:
        return 99.99
    return rate


def logit_of(rate):
    p = clamp_rate(rate) / 100.0
    return math.log(p / (1.0 - p))


def inv_logit(value):
    if value > 20.0:
        value = 20.0
    if value < -20.0:
        value = -20.0
    return 100.0 / (1.0 + math.exp(-value))


def norm_quantile(level):
    if level <= 0.0:
        return -8.0
    if level >= 1.0:
        return 8.0
    if level == 0.05:
        return Z_05
    if level == 0.25:
        return Z_25
    if level == 0.5:
        return 0.0
    if level == 0.75:
        return Z_75
    if level == 0.95:
        return Z_95
    lo = -8.0
    hi = 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        cdf = 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0)))
        if cdf < level:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def release_midnight(case):
    game = case.get("game") or {}
    raw = game.get("release_date")
    if raw:
        try:
            day = datetime.fromisoformat(raw)
            if day.tzinfo is None:
                day = day.replace(tzinfo=timezone.utc)
            return day
        except ValueError:
            pass
    first = case.get("history") or []
    if first:
        moment = parse_time(first[0]["at"])
        return moment.replace(hour=0, minute=0, second=0, microsecond=0)
    return datetime(2026, 11, 19, tzinfo=timezone.utc)


def log_time(moment, origin):
    hours = (moment - origin).total_seconds() / 3600.0
    if hours < -0.9:
        hours = -0.9
    return math.log(1.0 + hours)


def least_squares(xs, ys):
    n = len(xs)
    if n == 0:
        return 0.0, 0.0
    if n == 1:
        return ys[0], 0.0
    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    denom = sum((x - mean_x) * (x - mean_x) for x in xs)
    if denom == 0.0:
        return mean_y, 0.0
    slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
    return mean_y - slope * mean_x, slope


def fallback_row(trophy_id):
    return {"id": trophy_id, "q": [0.01, 0.01, 0.01, 0.01, 0.01]}


def quantiles_from(median, total_spread, levels):
    if median >= 100.0:
        return [100.0, 100.0, 100.0, 100.0, 100.0]
    if median <= 0.01:
        return [0.01, 0.01, 0.01, 0.01, 0.01]
    centre = logit_of(median)
    step = total_spread / (2.0 * 1.6448536269514722)
    out = []
    previous = 0.01
    for level in levels:
        value = half_up_2dp(inv_logit(centre + norm_quantile(level) * step))
        if value < 0.01:
            value = 0.01
        if value > 100.0:
            value = 100.0
        if value < previous:
            value = previous
        out.append(value)
        previous = value
    return out


def last_known_rate(history, index):
    for read in reversed(history):
        rate = read["rates"][index]
        if rate is not None:
            return rate
    return None


def find_cut(history):
    best_players_ratio = 1.0
    best_players_at = None
    best_rate_jump = 0.0
    best_rate_at = None
    for i in range(len(history) - 1):
        before = history[i]
        after = history[i + 1]
        players_before = before.get("players_tracked")
        players_after = after.get("players_tracked")
        if players_before and players_after and players_before > 0:
            ratio = players_after / players_before
            if ratio > best_players_ratio:
                best_players_ratio = ratio
                best_players_at = i
        rates_before = before["rates"]
        rates_after = after["rates"]
        for j in range(min(len(rates_before), len(rates_after))):
            if rates_before[j] is None or rates_after[j] is None:
                continue
            jump = abs(logit_of(rates_after[j]) - logit_of(rates_before[j]))
            if jump > best_rate_jump:
                best_rate_jump = jump
                best_rate_at = i
    if best_players_at is not None and best_players_ratio > JUMP_PLAYERS:
        return best_players_at + 1
    if best_rate_at is not None and best_rate_jump > JUMP_RATE:
        return best_rate_at + 1
    return None


def history_is_flat(history):
    if not history:
        return True
    first = history[0]
    for read in history[1:]:
        if read.get("players_tracked") != first.get("players_tracked"):
            return False
        if read["rates"] != first["rates"]:
            return False
    return True


def flat_forecast(case, levels):
    history = case["history"]
    last = history[-1]
    players = last.get("players_tracked")
    if not players:
        return [fallback_row(t["id"]) for t in case["trophies"]]
    target_players = players * REFRESH_TO / REFRESH_FROM
    newcomers = target_players - players
    if newcomers < 0.0:
        # The page already sits past the documented refresh level, so
        # there is no next wave of newcomers left to model.
        newcomers = 0.0
    last_rates = last["rates"]
    known = []
    for i, trophy in enumerate(case["trophies"]):
        rate = last_rates[i]
        if rate is None:
            rate = last_known_rate(history, i)
        known.append(rate)
    distinct = sorted({r for r in known if r is not None and r < 100.0}, reverse=True)
    top_rates = set(distinct[:3])
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        rate = known[i]
        if rate is None:
            rows.append(fallback_row(trophy["id"]))
            continue
        if rate >= 100.0:
            rows.append({"id": trophy["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        earned = half_up_int(rate / 100.0 * players)
        prefix = 2 if trophy["id"] <= PREFIX_COUNT else 0
        is_top = rate in top_rates
        bump = 0
        if is_top:
            bump = half_up_int(newcomers * rate / 100.0 * NEWCOMER_MID)
        total = earned + prefix + bump
        median = half_up_2dp(100.0 * total / target_players)
        if is_top:
            low_total = earned + prefix + half_up_int(
                newcomers * rate / 100.0 * NEWCOMER_LO
            )
            high_total = earned + prefix + half_up_int(
                newcomers * rate / 100.0 * NEWCOMER_HI
            )
            low = half_up_2dp(100.0 * low_total / target_players)
            high = half_up_2dp(100.0 * high_total / target_players)
            band = [low, median, median, median, high]
            band = [max(0.01, min(100.0, v)) for v in band]
            band.sort()
            rows.append({"id": trophy["id"], "q": band})
            continue
        noise = 1.0 / target_players
        spread = noise
        if spread < FLAT_SPREAD_MIN:
            spread = FLAT_SPREAD_MIN
        if spread > FLAT_SPREAD_MAX:
            spread = FLAT_SPREAD_MAX
        rows.append(
            {"id": trophy["id"], "q": quantiles_from(median, spread, levels)}
        )
    return rows


def moving_forecast(case, levels, segment):
    history = case["history"]
    target = parse_time(case["target"])
    horizon_days = case.get("horizon_minutes", 0) / 1440.0
    if horizon_days < 0.0:
        horizon_days = 0.0
    origin = release_midnight(case)
    target_x = log_time(target, origin)
    xs = [log_time(parse_time(read["at"]), origin) for read in segment]
    players_log = []
    players_ok = []
    for k, read in enumerate(segment):
        count = read.get("players_tracked")
        if count and count > 0:
            players_log.append(math.log(count))
            players_ok.append(k)
    if players_ok:
        fit_x = [xs[k] for k in players_ok]
        players_intercept, players_slope = least_squares(fit_x, players_log)
        players_pred = math.exp(players_intercept + players_slope * target_x)
    else:
        players_pred = None
    rows = []
    for i, trophy in enumerate(case["trophies"]):
        last_rate = last_known_rate(history, i)
        if last_rate is None:
            rows.append(fallback_row(trophy["id"]))
            continue
        if last_rate >= 100.0:
            rows.append({"id": trophy["id"], "q": [100.0, 100.0, 100.0, 100.0, 100.0]})
            continue
        tail = segment[-3:]
        tail_rates = [read["rates"][i] for read in tail if read["rates"][i] is not None]
        if len(tail_rates) >= 1:
            tail_logits = [logit_of(r) for r in tail_rates]
            wobble = max(tail_logits) - min(tail_logits)
        else:
            wobble = 0.0
        if wobble < FLAT_LOGIT_RANGE:
            median = half_up_2dp(tail_rates[-1])
            spread = MOVE_FLOOR + wobble
            if spread > MOVE_SPREAD_CAP:
                spread = MOVE_SPREAD_CAP
            rows.append(
                {"id": trophy["id"], "q": quantiles_from(median, spread, levels)}
            )
            continue
        earned_xs = []
        earned_log = []
        rate_seen = []
        for k, read in enumerate(segment):
            rate = read["rates"][i]
            count = read.get("players_tracked")
            if rate is None or not count or count <= 0:
                continue
            earned = half_up_int(rate / 100.0 * count)
            if earned < 1:
                earned = 1
            earned_xs.append(xs[k])
            earned_log.append(math.log(earned))
            rate_seen.append(rate)
        if len(earned_xs) < 2 or players_pred is None:
            median = half_up_2dp(last_rate)
            spread = MOVE_FLOOR + wobble
            if spread > MOVE_SPREAD_CAP:
                spread = MOVE_SPREAD_CAP
            rows.append(
                {"id": trophy["id"], "q": quantiles_from(median, spread, levels)}
            )
            continue
        earned_intercept, earned_slope = least_squares(earned_xs, earned_log)
        earned_pred = math.exp(earned_intercept + earned_slope * target_x)
        median_raw = 100.0 * earned_pred / players_pred
        if median_raw < 0.01:
            median_raw = 0.01
        if median_raw > 100.0:
            median_raw = 100.0
        median = half_up_2dp(median_raw)
        squared = 0.0
        counted = 0
        for k, read in enumerate(segment):
            rate = read["rates"][i]
            count = read.get("players_tracked")
            if rate is None or not count or count <= 0:
                continue
            fitted = 100.0 * math.exp(
                earned_intercept + earned_slope * xs[k]
            ) / math.exp(players_intercept + players_slope * xs[k])
            diff = logit_of(fitted) - logit_of(rate)
            squared += diff * diff
            counted += 1
        residual = math.sqrt(squared / counted) if counted else 0.0
        spread = MOVE_FLOOR + MOVE_HORIZON_RATE * math.sqrt(horizon_days) + residual
        if spread > MOVE_SPREAD_CAP:
            spread = MOVE_SPREAD_CAP
        rows.append(
            {"id": trophy["id"], "q": quantiles_from(median, spread, levels)}
        )
    return rows


def forecast_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    history = case.get("history") or []
    if not history:
        return [fallback_row(t["id"]) for t in case["trophies"]]
    cut = find_cut(history)
    if cut is None and history_is_flat(history):
        return flat_forecast(case, levels)
    if cut is None:
        segment = history
    else:
        segment = history[cut:]
    if not segment:
        segment = history
    return moving_forecast(case, levels, segment)


def run_case(path, out_dir):
    with open(path, "r", encoding="utf-8") as handle:
        case = json.load(handle)
    rows = forecast_case(case)
    stem = os.path.splitext(os.path.basename(path))[0]
    out_path = os.path.join(out_dir, stem + ".out")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"trophies": rows}, handle, separators=(",", ":"))
    return out_path


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    os.makedirs(out_dir, exist_ok=True)
    for path in sorted(glob.glob(os.path.join(in_dir, "*.in"))):
        run_case(path, out_dir)


if __name__ == "__main__":
    main()
