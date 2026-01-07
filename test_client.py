#!/usr/bin/env python3
"""
Test client for AI Duet FastAPI server.
This script demonstrates how to connect to the WebSocket API and interact with the duet system.
"""

import asyncio
import json
import websockets
import base64
import sys
from typing import Optional


async def test_duet_client():
    """Connect to the AI Duet server and test basic functionality."""

    # Configuration
    host = "localhost"
    port = 8000
    session_id = "test-session-123"

    websocket_url = f"ws://{host}:{port}/ws/{session_id}"

    print(f"Connecting to {websocket_url}...")

    try:
        async with websockets.connect(websocket_url) as websocket:
            print("✅ Connected to AI Duet server")

            # Wait for initial session state
            initial_msg = await websocket.recv()
            data = json.loads(initial_msg)

            if data["type"] == "session_state":
                print(f"\n📋 Session Info:")
                print(f"   Session ID: {data['session_id']}")
                print(f"   Agents: {list(data['agents'].keys())}")
                for agent_id, agent_info in data["agents"].items():
                    print(f"   {agent_id}: {agent_info['name']} ({agent_info['lang']})")

            # Test 1: Start the duet
            print("\n🚀 Test 1: Starting duet conversation...")
            await websocket.send(json.dumps({"type": "start_duet"}))

            # Listen for a few agent responses
            print("🎧 Listening to agent conversation...")
            for i in range(4):  # Listen for 4 messages (2 turns)
                try:
                    msg = await asyncio.wait_for(websocket.recv(), timeout=10.0)
                    data = json.loads(msg)

                    if data["type"] == "agent_text":
                        print(f"   {data['agent']} ({data['name']}): {data['text']}")
                    elif data["type"] == "agent_audio":
                        print(
                            f"   {data['agent']} sent audio (length: {len(data['wav_b64'])} chars)"
                        )
                    elif data["type"] == "ok":
                        print(f"   Server: {data['what']}")
                    else:
                        print(f"   Received: {data['type']}")

                except asyncio.TimeoutError:
                    print("   Timeout waiting for response")
                    break

            # Test 2: Send text interruption
            print("\n💬 Test 2: Sending text interruption...")
            test_message = "Hello agents! What are you discussing?"
            await websocket.send(
                json.dumps({"type": "user_text", "text": test_message})
            )

            # Wait for acknowledgement
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                data = json.loads(msg)
                if data["type"] == "user_text_ack":
                    print(f"   ✅ Message acknowledged: {data['text']}")
            except asyncio.TimeoutError:
                print("   ⏰ No acknowledgement received")

            # Test 3: Stop the duet
            print("\n⏹️ Test 3: Stopping duet...")
            await websocket.send(json.dumps({"type": "stop"}))

            # Wait for confirmation
            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(msg)
                if data["type"] == "ok" and data["what"] == "stopped":
                    print("   ✅ Duet stopped successfully")
            except asyncio.TimeoutError:
                print("   ⏰ No stop confirmation received")

            # Test 4: Reset session
            print("\n🔄 Test 4: Resetting session...")
            await websocket.send(json.dumps({"type": "reset"}))

            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(msg)
                if data["type"] == "ok" and data["what"] == "reset":
                    print("   ✅ Session reset successfully")
            except asyncio.TimeoutError:
                print("   ⏰ No reset confirmation received")

            # Test 5: Update agent settings
            print("\n⚙️ Test 5: Updating Agent A instructions...")
            new_instructions = "You are now a poet. Speak in rhymes about technology."
            await websocket.send(
                json.dumps(
                    {
                        "type": "set_agent",
                        "agent": "A",
                        "instructions": new_instructions,
                    }
                )
            )

            try:
                msg = await asyncio.wait_for(websocket.recv(), timeout=3.0)
                data = json.loads(msg)
                if data["type"] == "ok" and data["what"] == "agent_updated":
                    print("   ✅ Agent A updated successfully")
            except asyncio.TimeoutError:
                print("   ⏰ No update confirmation received")

            print("\n🎉 All tests completed!")
            print("\n📝 Summary of tested operations:")
            print("   1. Started duet conversation")
            print("   2. Listened to agent responses")
            print("   3. Sent text interruption")
            print("   4. Stopped duet")
            print("   5. Reset session")
            print("   6. Updated agent instructions")

    except ConnectionRefusedError:
        print(f"❌ Could not connect to {websocket_url}")
        print(
            "   Make sure the server is running with: uvicorn main:app --reload --host 0.0.0.0 --port 8000"
        )
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


async def interactive_client():
    """Interactive client for manual testing."""

    host = "localhost"
    port = 8000
    session_id = "interactive-" + str(int(time.time()))

    websocket_url = f"ws://{host}:{port}/ws/{session_id}"

    print(f"Interactive AI Duet Client")
    print(f"Connecting to {websocket_url}...")

    try:
        async with websockets.connect(websocket_url) as websocket:
            print("✅ Connected!")

            # Get initial session state
            initial_msg = await websocket.recv()
            data = json.loads(initial_msg)
            if data["type"] == "session_state":
                print(f"\nSession: {data['session_id']}")
                for agent_id, agent_info in data["agents"].items():
                    print(
                        f"{agent_id}: {agent_info['name']} - {agent_info['instructions'][:50]}..."
                    )

            print("\nAvailable commands:")
            print("  start     - Start duet conversation")
            print("  stop      - Stop duet")
            print("  text <msg>- Send text message")
            print("  reset     - Reset session")
            print("  agents    - Show agent info")
            print("  update <agent> <instructions> - Update agent")
            print("  quit      - Exit")

            while True:
                try:
                    command = input("\n> ").strip().lower()

                    if command == "quit":
                        print("Goodbye!")
                        break

                    elif command == "start":
                        await websocket.send(json.dumps({"type": "start_duet"}))
                        print("Started duet")

                    elif command == "stop":
                        await websocket.send(json.dumps({"type": "stop"}))
                        print("Stopped duet")

                    elif command.startswith("text "):
                        message = command[5:].strip()
                        if message:
                            await websocket.send(
                                json.dumps({"type": "user_text", "text": message})
                            )
                            print(f"Sent: {message}")

                    elif command == "reset":
                        await websocket.send(json.dumps({"type": "reset"}))
                        print("Reset session")

                    elif command == "agents":
                        await websocket.send(json.dumps({"type": "session_state"}))

                    elif command.startswith("update "):
                        parts = command[7:].split(" ", 1)
                        if len(parts) == 2:
                            agent, instructions = parts
                            await websocket.send(
                                json.dumps(
                                    {
                                        "type": "set_agent",
                                        "agent": agent.upper(),
                                        "instructions": instructions,
                                    }
                                )
                            )
                            print(f"Updating {agent}...")

                    else:
                        print("Unknown command. Type 'quit' to exit.")

                    # Check for any incoming messages
                    try:
                        while True:
                            msg = await asyncio.wait_for(websocket.recv(), timeout=0.1)
                            data = json.loads(msg)

                            if data["type"] == "agent_text":
                                print(f"\n{data['agent']}: {data['text']}")
                            elif data["type"] == "user_text_ack":
                                print(f"\n✓ Message acknowledged")
                            elif data["type"] == "ok":
                                print(f"\n✓ {data['what']}")
                            elif data["type"] == "error":
                                print(f"\n✗ Error: {data['message']}")

                    except asyncio.TimeoutError:
                        pass  # No more messages

                except KeyboardInterrupt:
                    print("\nInterrupted. Sending stop command...")
                    await websocket.send(json.dumps({"type": "stop"}))
                    break

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    import time

    print("AI Duet Test Client")
    print("===================")
    print("1. Run automated tests")
    print("2. Interactive mode")
    print("3. Exit")

    choice = input("\nSelect option (1-3): ").strip()

    if choice == "1":
        asyncio.run(test_duet_client())
    elif choice == "2":
        asyncio.run(interactive_client())
    else:
        print("Goodbye!")
