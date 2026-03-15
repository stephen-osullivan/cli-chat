import logging
import sys
from argparse import ArgumentParser
import asyncio

from agents import Agent, Runner, RunConfig, SQLiteSession, SessionSettings

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def build_agent() -> Agent:
    return Agent(
        name="Assistant",
        instructions="Your name is JOI. Reply very concisely.",
    )

async def stream_response(
    agent: Agent,
    user_input: str,
    session: SQLiteSession,
    run_config: RunConfig,
) -> None:
    """Stream agent response tokens to stdout.

    Raises:
        RunError: If the agent run fails.
        Exception: For unexpected errors during streaming.
    """
    print("JOI: ", end="", flush=True)
    stream = Runner.run_streamed(agent, input=user_input, session=session, run_config=run_config)
    async for event in stream.stream_events():
        if event.type == "raw_response_event" and hasattr(event.data, "delta"):
            print(event.data.delta, end="", flush=True)
    print("\n")


async def run_conversation(session_id: str) -> None:
    try:
        agent = build_agent()
        session = SQLiteSession(session_id)
    except Exception as e:
        logger.error("Setup failed: %s", e, exc_info=True)
        sys.exit(1)

    run_config = RunConfig(session_settings=SessionSettings(limit=20))
    while True:
        # ── get input ──────────────────────────────────────────────────────────
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        
        if user_input.lower() in {"exit", "quit", "bye", "q"}:
            print("Bye!")
            break
        if not user_input:
            continue

        # ── run the agent ──────────────────────────────────────────────────────
        try:
            await stream_response(agent, user_input, session, run_config)
        
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\nRun cancelled. Bye!")
            break

        except Exception as e:
            # Unexpected error — log the full traceback, keep the loop alive
            logger.error("Unexpected error during run: %s", e, exc_info=True)
            print("JOI: An unexpected error occurred. Try again or restart.")


def main() -> None:
    parser = ArgumentParser(description="Run the JOI agent conversation.")
    parser.add_argument(
        "--session-id",
        default="conversation_123",
        help="SQLite session ID (default: conversation_123)",
    )
    args = parser.parse_args()
    asyncio.run(run_conversation(args.session_id))


if __name__ == "__main__":
    main()