"""100人の村でリーダーと権力集中が生まれる条件を探る教育用ABM。

必要: numpy, matplotlib
実行: python village_power_sim.py --runs 1000
注: 数値・意思決定ルールは仮定であり、現実の確率の推定ではない。
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

N_PEOPLE = 100
TERM = 12  # 一時的に指揮を委任する期間


@dataclass(frozen=True)
class Scenario:
    name: str
    crisis_rate: float = 0.12       # 委任期間の各回で危機が起きる確率
    recall: float = 0.05           # 解任・情報共有制度の強さ (0〜1)
    strong_person: bool = False    # 1人だけ身体能力を3倍にする


def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))


def simulate_one(rng: np.random.Generator, s: Scenario) -> tuple[bool, bool, bool]:
    """返り値: (リーダー誕生, 強い人が選ばれた, 期限後の権力保持に成功)。"""
    strength = np.ones(N_PEOPLE)  # 初期の能力・財産・地位は全員等しい
    if s.strong_person:
        strength[0] = 3.0

    # ① 自発的に現れた提案を見て、各自が「誰についていくか」を決める。
    #    提案の目立ち方は偶然（特定の人の生まれつきの才能ではない）。
    initiative = rng.normal(0, 0.8, N_PEOPLE)
    perceived = (
        initiative[None, :] + 0.75 * (strength[None, :] - 1)
        + rng.normal(0, 1.5, (N_PEOPLE, N_PEOPLE))
    )
    followers = np.bincount(np.argmax(perceived, axis=1), minlength=N_PEOPLE)
    leader = int(followers.argmax())
    leader_emerged = followers[leader] >= 10  # 10人以上が同じ人を自発的に支持
    if not leader_emerged:
        return False, False, False

    strong_won = s.strong_person and leader == 0

    # ② 期間限定で、リーダーに共同備蓄の管理を任せる。
    #    危機のたびに管理下の備蓄が増える（仮定）。
    controlled_resources = 0.0
    for _ in range(TERM):
        crisis = rng.random() < s.crisis_rate
        controlled_resources += 0.6 + (2.5 if crisis else 0.0)

    # ③ 管理資源から特別待遇を出し、支持者を集められる。
    #    全員「個人支配は嫌」だが、職務上の報酬で受諾する人もいる。
    offers = min(
        N_PEOPLE - 1,
        round((4 + controlled_resources * 1.2) * (1 - 0.5 * s.recall)),
    )
    candidates = rng.choice(np.delete(np.arange(N_PEOPLE), leader), offers, replace=False)
    accepted = rng.random(len(candidates)) < sigmoid(-0.1 + 0.025 * controlled_resources)
    supporters = int(accepted.sum())

    # ④ 任期満了。個人的利益と制度上の制約に応じて延長を試みる。
    p_overstay = sigmoid(-1.3 + 0.07 * controlled_resources - 2.5 * s.recall)
    if rng.random() >= p_overstay:
        return True, bool(strong_won), False

    # ⑤ 人々は延長に反対する。ただし公開で抵抗できるかは制度次第。
    #    反対意思は共有されていても、協調できるとは限らない。
    p_mobilize = sigmoid(-2.4 + 4.5 * s.recall)
    public_opposition = rng.binomial(N_PEOPLE - 1 - supporters, p_mobilize)

    # ⑥ 公開の反対が支持者の組織力を上回れば解任。そうでなければ居座り成功。
    entrenched = supporters >= public_opposition
    return True, bool(strong_won), bool(entrenched)


def experiment(runs: int, seed: int) -> list[dict[str, float | str]]:
    scenarios = [
        Scenario("Equal / low crisis", crisis_rate=0.12, recall=0.05),
        Scenario("One strong person", crisis_rate=0.12, recall=0.05, strong_person=True),
        Scenario("Frequent crises", crisis_rate=0.65, recall=0.05),
        Scenario("Strong recall rules", crisis_rate=0.12, recall=0.80),
    ]
    results: list[dict[str, float | str]] = []
    for scenario_index, s in enumerate(scenarios):
        # シナリオ間の乱数系列を分け、実行結果を再現可能にする。
        rng = np.random.default_rng(seed + 10000 * scenario_index)
        outcomes = np.asarray([simulate_one(rng, s) for _ in range(runs)])
        results.append({
            "scenario": s.name,
            "leader_emergence_percent": 100 * outcomes[:, 0].mean(),
            "strong_person_led_percent": 100 * outcomes[:, 1].mean(),
            "power_retention_percent": 100 * outcomes[:, 2].mean(),
        })
    return results


def save_outputs(results: list[dict[str, float | str]], destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    csv_path = destination / "village_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)

    labels = [str(item["scenario"]) for item in results]
    values = [float(item["power_retention_percent"]) for item in results]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.barh(labels[::-1], values[::-1], color="steelblue")
    ax.set_xlim(0, 100)
    ax.set_xlabel("Power retained after temporary mandate (%)")
    ax.set_title("Toy village model: outcomes depend on assumptions")
    for bar in bars:
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f"{bar.get_width():.1f}%", va="center")
    ax.set_xlim(0, max(100, max(values) + 9))
    fig.tight_layout()
    fig.savefig(destination / "village_power_chart.png", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    args = parser.parse_args()
    if args.runs < 1:
        parser.error("--runs must be >= 1")
    results = experiment(args.runs, args.seed)
    save_outputs(results, args.output_dir)
    print(f"{args.runs} trials per scenario (seed={args.seed})")
    print(f"{'Scenario':<22} {'Leader':>10} {'Strong led':>12} {'Kept power':>13}")
    for r in results:
        print(f"{r['scenario']:<22} {r['leader_emergence_percent']:>9.1f}% "
              f"{r['strong_person_led_percent']:>11.1f}% "
              f"{r['power_retention_percent']:>12.1f}%")
    print(f"Wrote {args.output_dir / 'village_results.csv'}")
    print(f"Wrote {args.output_dir / 'village_power_chart.png'}")


if __name__ == "__main__":
    main()