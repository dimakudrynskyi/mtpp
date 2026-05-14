"""
End-to-end автоматичний тест чат-сервера.

Запускає сервер у фоні, підключає кілька TCP-клієнтів, виконує всі
основні сценарії, перевіряє коректність відповідей і фінальну статистику.
"""
from __future__ import annotations
import base64
import json
import socket
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from server import start_servers


class TestClient:
    """Простий синхронний клієнт для тестів."""

    def __init__(self, username: str, host: str = "127.0.0.1", port: int = 9000):
        self.username = username
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((host, port))
        self.sock.settimeout(5.0)
        self.received: list[dict] = []
        self.alive = True
        self.recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        self.recv_thread.start()
        self.send({"type": "register", "username": username})

    def _recv_loop(self):
        buffer = b""
        try:
            while self.alive:
                try:
                    chunk = self.sock.recv(4096)
                except socket.timeout:
                    continue
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.strip():
                        self.received.append(json.loads(line.decode("utf-8")))
        except Exception:
            pass

    def send(self, msg: dict):
        line = json.dumps(msg) + "\n"
        self.sock.sendall(line.encode("utf-8"))

    def close(self):
        self.alive = False
        try:
            # shutdown(SHUT_RDWR) надсилає TCP FIN, серверний recv() поверне 0
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass

    def filter(self, msg_type: str) -> list[dict]:
        return [m for m in self.received if m.get("type") == msg_type]

    def wait_for(self, msg_type: str, timeout: float = 2.0) -> dict | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            msgs = self.filter(msg_type)
            if msgs:
                return msgs[-1]
            time.sleep(0.001)   # 1ms polling — для точних latency-вимірів
        return None


# ============================================================================
# Test scenarios
# ============================================================================

def test_register_and_welcome(state):
    print("\n[T1] Register & welcome")
    alice = TestClient("alice_t1")
    welcome = alice.wait_for("welcome")
    assert welcome is not None, "no welcome message"
    assert welcome["username"] == "alice_t1"
    print("  ✓ welcome received")
    alice.close()
    time.sleep(0.2)


def test_private_messaging(state):
    print("\n[T2] Private messaging")
    alice = TestClient("alice_t2")
    bob = TestClient("bob_t2")
    alice.wait_for("welcome")
    bob.wait_for("welcome")

    alice.send({"type": "private", "to": "bob_t2", "text": "Hi Bob!"})
    msg = bob.wait_for("user_msg")
    assert msg is not None and msg["text"] == "Hi Bob!", f"unexpected msg: {msg}"
    print("  ✓ private delivered")

    delivered = alice.wait_for("delivered")
    assert delivered and delivered.get("online") == True
    print("  ✓ delivery confirmation")

    alice.close()
    bob.close()
    time.sleep(0.2)


def test_broadcast(state):
    print("\n[T3] Broadcast")
    a = TestClient("a_t3"); b = TestClient("b_t3"); c = TestClient("c_t3")
    a.wait_for("welcome"); b.wait_for("welcome"); c.wait_for("welcome")
    time.sleep(0.2)

    a.send({"type": "broadcast", "text": "Hello everyone!"})
    bm = b.wait_for("broadcast_msg")
    cm = c.wait_for("broadcast_msg")
    assert bm and bm["text"] == "Hello everyone!"
    assert cm and cm["text"] == "Hello everyone!"
    print(f"  ✓ broadcast received by 2 clients")

    a.close(); b.close(); c.close()
    time.sleep(0.2)


def test_groups(state):
    print("\n[T4] Groups")
    a = TestClient("a_t4"); b = TestClient("b_t4"); c = TestClient("c_t4")
    a.wait_for("welcome"); b.wait_for("welcome"); c.wait_for("welcome")

    a.send({"type": "group_create", "group_name": "devs_t4"})
    b.send({"type": "group_join", "group_name": "devs_t4"})
    c.send({"type": "group_join", "group_name": "devs_t4"})
    time.sleep(0.3)

    a.send({"type": "group_msg", "group_name": "devs_t4", "text": "stand-up at 10!"})
    bm = b.wait_for("group_msg")
    cm = c.wait_for("group_msg")
    assert bm and bm.get("group") == "devs_t4" and bm["text"] == "stand-up at 10!"
    assert cm and cm.get("group") == "devs_t4"
    print("  ✓ group message delivered to 2 members")

    a.close(); b.close(); c.close()
    time.sleep(0.2)


def test_offline_messages(state):
    print("\n[T5] Offline messages")
    alice = TestClient("alice_t5")
    alice.wait_for("welcome")
    alice.send({"type": "private", "to": "ghost_t5", "text": "you online?"})
    delivered = alice.wait_for("delivered")
    assert delivered and delivered.get("stored_offline") == True
    print("  ✓ message stored offline")

    # ghost_t5 reconnects
    ghost = TestClient("ghost_t5")
    ghost.wait_for("welcome")
    msg = ghost.wait_for("user_msg")
    assert msg and msg.get("offline") == True and msg["text"] == "you online?"
    print("  ✓ offline message delivered on reconnect")

    alice.close(); ghost.close()
    time.sleep(0.2)


def test_history(state):
    print("\n[T6] Message history")
    a = TestClient("a_t6"); b = TestClient("b_t6")
    a.wait_for("welcome"); b.wait_for("welcome")

    for i in range(5):
        a.send({"type": "private", "to": "b_t6", "text": f"msg {i}"})
        time.sleep(0.05)
    time.sleep(0.3)

    a.send({"type": "history", "with_user": "b_t6"})
    hist = a.wait_for("history")
    assert hist and len(hist["messages"]) == 5
    print(f"  ✓ history returned {len(hist['messages'])} messages")
    a.close(); b.close()
    time.sleep(0.2)


def test_files(state):
    print("\n[T7] File transfer")
    a = TestClient("a_t7"); b = TestClient("b_t7")
    a.wait_for("welcome"); b.wait_for("welcome")

    payload = b"Hello, world! This is a test file content."
    data_b64 = base64.b64encode(payload).decode("ascii")
    a.send({"type": "file_send", "to": "b_t7", "filename": "test.txt", "data_b64": data_b64})

    file_msg = b.wait_for("file_msg")
    assert file_msg and file_msg["filename"] == "test.txt"
    received = base64.b64decode(file_msg["data_b64"])
    assert received == payload
    print(f"  ✓ file received intact ({len(payload)} bytes)")
    a.close(); b.close()
    time.sleep(0.2)


def test_disconnect_detection(state):
    print("\n[T8] Disconnect detection")
    a = TestClient("a_t8"); b = TestClient("b_t8")
    a.wait_for("welcome"); b.wait_for("welcome")
    time.sleep(0.2)

    # b disconnects abruptly
    b.close()
    time.sleep(0.5)

    a.send({"type": "list_users"})
    users = a.wait_for("users_list", timeout=3.0)
    assert users and "b_t8" not in users["users"]
    print(f"  ✓ disconnect detected; users now: {users['users']}")

    # Should also have got a user_left message
    left = a.wait_for("user_left", timeout=2.0)
    assert left and left["username"] == "b_t8"
    print(f"  ✓ user_left notification received")
    a.close()
    time.sleep(0.2)


def test_concurrent_load(state):
    """Стрес-тест: 30 клієнтів одночасно spam'ять broadcast'ами."""
    print("\n[T9] Concurrent load")
    N = 30
    clients = []
    for i in range(N):
        clients.append(TestClient(f"loadtest_{i}"))
        time.sleep(0.005)
    for c in clients:
        c.wait_for("welcome", timeout=3.0)
    time.sleep(0.3)
    print(f"  {N} clients connected")

    # Кожен надсилає 5 broadcast'ів
    M = 5
    t0 = time.perf_counter()
    for c in clients:
        for j in range(M):
            c.send({"type": "broadcast", "text": f"msg{j} from {c.username}"})
    # Зачекати трошки на доставку
    time.sleep(2.0)
    elapsed = time.perf_counter() - t0
    expected = N * M * (N - 1)   # кожен броадкаст → (N-1) доставок
    actual = sum(len(c.filter("broadcast_msg")) for c in clients)
    print(f"  {N*M} broadcasts sent → {actual}/{expected} deliveries  ({elapsed:.2f}s)")
    assert actual >= expected * 0.95, f"Too many lost: {actual}/{expected}"
    print(f"  ✓ delivery rate: {100 * actual / expected:.1f}%")

    for c in clients:
        c.close()
    time.sleep(0.5)


def test_thread_safety(state):
    """Стрес-тест на race: 20 потоків одночасно реєструють однакове ім'я."""
    print("\n[T10] Thread safety on register")
    succeeded = []
    failed = []

    def try_register():
        try:
            c = TestClient("contested_user")
            time.sleep(0.3)
            errs = c.filter("error")
            wel = c.filter("welcome")
            if wel:
                succeeded.append(c.username)
            elif errs:
                failed.append(errs[0]["code"])
            c.close()
        except Exception as e:
            failed.append(str(e))

    threads = [threading.Thread(target=try_register) for _ in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()

    print(f"  succeeded: {len(succeeded)}, failed: {len(failed)}")
    # Має бути РІВНО 1 успішних реєстрацій (через RLock)
    assert len(succeeded) == 1, f"Race: {len(succeeded)} succeeded instead of 1"
    print(f"  ✓ exactly 1 registration succeeded — locking works")
    time.sleep(0.5)


# ============================================================================
# Main
# ============================================================================

def main():
    print("Starting servers...")
    state, stop = start_servers(tcp_port=9000, ws_port=9001)
    time.sleep(1.0)

    tests = [
        test_register_and_welcome,
        test_private_messaging,
        test_broadcast,
        test_groups,
        test_offline_messages,
        test_history,
        test_files,
        test_disconnect_detection,
        test_thread_safety,
        test_concurrent_load,
    ]

    passed = 0
    failed = 0
    failures = []
    for test in tests:
        try:
            test(state)
            passed += 1
        except AssertionError as e:
            failed += 1
            failures.append((test.__name__, str(e)))
            print(f"  ✗ FAILED: {e}")
        except Exception as e:
            failed += 1
            failures.append((test.__name__, repr(e)))
            print(f"  ✗ ERROR: {e}")

    # Final stats
    final_stats = state.get_stats()
    print("\n" + "=" * 60)
    print(f"RESULT: {passed}/{len(tests)} passed, {failed} failed")
    if failures:
        print("Failures:")
        for name, err in failures:
            print(f"  - {name}: {err}")
    print("\nFinal server stats:")
    for k, v in final_stats.items():
        print(f"  {k}: {v}")

    # Save stats for the report
    Path("results").mkdir(exist_ok=True)
    import json as _json
    Path("results/test_stats.json").write_text(
        _json.dumps({"passed": passed, "failed": failed, "stats": final_stats},
                     indent=2, ensure_ascii=False))

    stop.set()
    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
