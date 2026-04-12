"""Evaluation-oriented inference script for the price negotiation environment.""""""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME 
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
    
- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 rewards=0.00,0.00,1.00
"""

from __future__ import annotations

import argparse
import asyncio
import time
import sys
from pathlib import Path
from typing import Optional
import json

from server.price_negotiation_environment import Difficulty

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_PARENT = PACKAGE_ROOT.parent
if str(PACKAGE_PARENT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_PARENT))

from price_negotiation import PriceNegotiationAction, PriceNegotiationEnv
from price_negotiation.reward import reward_breakdown, score_trajectory
from price_negotiation.rollout import initial_buyer_message, latest_seller_reply
from price_negotiation.server.helper_functions import DEFAULT_OPENAI_MODEL, get_openai_response
from price_negotiation.trajectory_types import TrajectoryResult, TrajectoryStep

import os


API_KEY = os.getenv("API_KEY") or os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("BUYER_MODEL") or os.getenv("MODEL_NAME") or DEFAULT_OPENAI_MODEL
IMAGE_NAME = os.getenv("IMAGE_NAME") or os.getenv("LOCAL_IMAGE_NAME") or "openenv-price_negotiation"
ENV_BASE_URL = os.getenv("ENV_BASE_URL") or "https://viditostwal-price-negotiation.hf.space"
TASK_NAME = os.getenv("TASK_NAME") or "price-negotiation"
BENCHMARK = os.getenv("BENCHMARK") or "price_negotiation"
TEMPERATURE = float(os.getenv("BUYER_TEMPERATURE") or "0.7")
SUCCESS_SCORE_THRESHOLD = float(os.getenv("SUCCESS_SCORE_THRESHOLD") or "0.4")
DEBUG = os.getenv("DEBUG", "").lower() in {"1", "true", "yes", "on"}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for inference."""
    parser = argparse.ArgumentParser(description="Run a price negotiation rollout.")
    parser.add_argument(
        "--id",
        help="Task or episode identifier to include in logs and forward on reset.",
    )
    parser.add_argument(
        "-d",
        "--difficulty",
        choices=("easy", "medium", "hard"),
        help="Difficulty of the sampled negotiation scenario.",
    )
    return parser.parse_args()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action[:10]} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(task: str, success: bool, steps: int, score: float, rewards: list[float]) -> None:
    try:
        rewards_str = json.dumps(rewards, separators=(",", ":"))
    except TypeError:
        rewards_str = str(rewards)

    print(
        f"[END] task={task} success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def debug_print(message: str) -> None:
    if DEBUG:
        print(message, flush=True)


def _generate_buyer_response(state) -> str:
    """Mirror rollout.py: canned opener first, model-backed turns afterwards."""
    buyer_response = initial_buyer_message(state)
    if buyer_response is None:
        buyer_response = get_openai_response(
            state.buyer_messages,
            model=MODEL_NAME,
            temperature=TEMPERATURE,
        )
    return buyer_response


def _build_trajectory(initial_observation, final_state, steps: list[TrajectoryStep]) -> TrajectoryResult:
    return TrajectoryResult(
        episode_id=final_state.episode_id,
        initial_observation=initial_observation,
        final_state=final_state.model_copy(deep=True),
        steps=steps,
    )


def _docker_env_vars() -> dict[str, str]:
    env_vars: dict[str, str] = {}
    if API_KEY:
        env_vars["API_KEY"] = API_KEY
    if API_BASE_URL:
        env_vars["API_BASE_URL"] = API_BASE_URL
    return env_vars


async def _connect_env():
    # if IMAGE_NAME:
    #     debug_print(f"[DEBUG] starting docker image: {IMAGE_NAME}")
    #     # debug_print(f"[DEBUG] docker env vars: {_docker_env_vars()}")
    #     env = await PriceNegotiationEnv.from_docker_image(
    #         IMAGE_NAME,
    #         env_vars=_docker_env_vars(),
    #     )
    #     debug_print("[DEBUG] docker container reported ready; waiting 10s before use")
    #     await asyncio.sleep(10)
    #     debug_print("[DEBUG] docker-backed env client ready")
    #     return env
    debug_print(f"[DEBUG] connecting to running env at: {ENV_BASE_URL}")
    env = PriceNegotiationEnv(base_url=ENV_BASE_URL)
    await env.connect()
    return env


async def main() -> None:
    args = parse_args()
    env = None
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    reward_breakdown_score = None
    task_id = args.id or args.difficulty or TASK_NAME

    

    for task in ['easy', 'medium', 'hard']:
        task_id = task
        diff = task

        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

        try:
            debug_print("[DEBUG] entering main rollout try block")
            env = await _connect_env()
            debug_print("[DEBUG] env connection object created")
            debug_print(f"[DEBUG] env={env}")
            debug_print("[DEBUG] env connected")
            reset_result = await env.reset(
                difficulty=diff
            )
            debug_print(f"[DEBUG] reset_result={reset_result}")
            state = await env.state()
            debug_print(f"[DEBUG] state={state}")
            debug_print(
                f"[DEBUG] reset complete: episode_id={state.episode_id} step_count={state.step_count}"
            )
            debug_print(f"[DEBUG] requested difficulty={diff}")
            debug_print(
                f"[DEBUG] sampled product: {state.product_info.get('product', {}).get('name', 'unknown')}"
            )
            trajectory_steps: list[TrajectoryStep] = []
            turn_limit = state.product_info.get("metadata", {}).get("max_turns")
            debug_print(f"[DEBUG] turn_limit={turn_limit}")

            while True:
                if turn_limit is not None and state.step_count >= turn_limit:
                    debug_print("[DEBUG] stopping rollout: reached turn limit")
                    break

                step_number = state.step_count + 1

                try:
                    debug_print(f"[DEBUG] generating buyer response for step={step_number}")
                    buyer_response = _generate_buyer_response(state)
                    debug_print(f"[DEBUG] buyer response step={step_number}: {buyer_response}")
                    step_result = await env.step(PriceNegotiationAction(buyer_response=buyer_response))
                    state = await env.state()
                    seller_reply = latest_seller_reply(state)
                    done = bool(step_result.done)

                    trajectory_steps.append(TrajectoryStep(buyer_response=buyer_response,observation=step_result.observation,state=state.model_copy(deep=True),seller_reply=seller_reply))
                    trajectory = _build_trajectory(initial_observation=reset_result.observation,final_state=state,steps=trajectory_steps)
                    score = score_trajectory(trajectory)

                    rewards.append(score)
                    steps_taken = step_number

                    log_step(step=step_number,action=buyer_response,reward=score,done=done,error=None)
                    debug_print(f"[DEBUG] step={step_number} seller_reply={seller_reply}")
                    debug_print(f"[DEBUG] step={step_number} status={step_result.observation.deal_status} reward={score} done={done}")

                    if done:
                        debug_print(f"[DEBUG] stopping rollout: env done at step={step_number}")
                        break

                except Exception as exc:
                    error_message = str(exc) or "unknown-error"
                    debug_print(f"[DEBUG] step exception at step={step_number}: {error_message}")
                    rewards.append(0.0)
                    steps_taken = step_number
                    log_step(step=step_number,action=locals().get("buyer_response", ""),reward=0.0,done=True,error=error_message)
                    break

            trajectory = _build_trajectory(initial_observation=reset_result.observation,final_state=state,steps=trajectory_steps)
            debug_print(f"[DEBUG] built trajectory with {len(trajectory_steps)} steps; computing reward")
            reward_breakdown_score = reward_breakdown(trajectory)
            score = score_trajectory(trajectory)
            success = score >= SUCCESS_SCORE_THRESHOLD
            debug_print(f"[DEBUG] final score={score} success={success}")
        except Exception as exc:
            debug_print(f"[DEBUG] outer exception: {exc}")
            success = False
        finally:
            if env is not None:
                try:
                    debug_print("[DEBUG] closing env")
                    await env.close()
                except Exception as exc:
                    debug_print(f"[DEBUG] env.close() exception: {exc}")
                    pass
            debug_print("[DEBUG] emitting final log line")
            log_end(task=task_id, success=success, steps=steps_taken, score=score, rewards=reward_breakdown_score)


if __name__ == "__main__":
    asyncio.run(main())
