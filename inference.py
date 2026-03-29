"""Run a local inference rollout against the price negotiation server."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from price_negotiation.reward import reward_breakdown, score_trajectory
from price_negotiation.rollout import format_product_name, run_rollout


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-url",
        default="http://localhost:8000",
        help="Base URL of the local negotiation server",
    )
    parser.add_argument(
        "--buyer-model",
        default="gpt-4.1-mini",
        help="OpenAI model used for buyer-side generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature for buyer-side generation",
    )
    parser.add_argument(
        "--max-turns",
        type=int,
        default=None,
        help="Optional hard limit on number of buyer steps",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print the full trajectory details",
    )
    args = parser.parse_args()

    trajectory = run_rollout(
        base_url=args.base_url,
        buyer_model=args.buyer_model,
        temperature=args.temperature,
        max_turns=args.max_turns,
    )

    if args.verbose:
        print(f"Episode ID: {trajectory.episode_id}")
        print(f"Product: {format_product_name(trajectory.final_state)}")
        print(f"Initial next turn: {trajectory.initial_observation.next_turn}")
        print(f"Initial deal status: {trajectory.initial_observation.deal_status}")
        print()

        for step in trajectory.steps:
            print(f"Buyer: {step.buyer_response}")
            if step.seller_reply:
                print(f"Seller: {step.seller_reply}")
            print(
                "Observation:",
                {
                    "next_turn": step.observation.next_turn,
                    "negotiation_round": step.observation.negotiation_round,
                    "deal_status": step.observation.deal_status,
                    "done": step.observation.done,
                },
            )
            print()

    breakdown = reward_breakdown(trajectory)
    print("Reward breakdown:", breakdown)
    print(f"Trajectory reward: {score_trajectory(trajectory):.4f}")


if __name__ == "__main__":
    main()
