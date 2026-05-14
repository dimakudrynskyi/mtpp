"""
CLI-клієнт чат-системи. Підтримує всі команди.

Використання:
  python3 cli_client.py <username> [host] [port]

Команди в чаті:
  /msg <user> <text>          приватне повідомлення
  /all <text>                 broadcast всім
  /users                      список онлайн
  /group new <name>           створити групу
  /group join <name>          приєднатись
  /group leave <name>         покинути
  /group msg <name> <text>    повідомлення в групу
  /groups                     список груп
  /history <user>             історія розмови
  /file <user> <path>         надіслати файл
  /typing <user>              позначити «друкує...»
  /stats                      статистика сервера
  /quit                       вийти
"""
from __future__ import annotations
import base64
import json
import socket
import sys
import threading
from pathlib import Path


class ChatClient:
    def __init__(self, username: str, host: str = "127.0.0.1", port: int = 9000):
        self.username = username
        self.host = host
        self.port = port
        self.sock: socket.socket | None = None
        self.alive = True

    def connect(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.host, self.port))
        self._send({"type": "register", "username": self.username})

    def _send(self, msg: dict):
        line = json.dumps(msg, ensure_ascii=False) + "\n"
        self.sock.sendall(line.encode("utf-8"))

    def _recv_loop(self):
        """Окремий потік, який читає вхідні повідомлення і виводить їх."""
        buffer = b""
        try:
            while self.alive:
                chunk = self.sock.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                        self._display(msg)
                    except Exception as e:
                        print(f"[parse error] {e}", file=sys.stderr)
        except Exception as e:
            print(f"[connection lost] {e}", file=sys.stderr)
        finally:
            self.alive = False

    def _display(self, msg: dict):
        t = msg.get("type")
        if t == "welcome":
            print(f"[server] welcome, {msg['username']}!")
        elif t == "user_msg":
            offline_mark = " [offline]" if msg.get("offline") else ""
            print(f"[{msg['from']} → me]{offline_mark}: {msg['text']}")
        elif t == "broadcast_msg":
            print(f"[{msg['from']} → ALL]: {msg['text']}")
        elif t == "group_msg":
            offline_mark = " [offline]" if msg.get("offline") else ""
            print(f"[{msg['from']} @ {msg.get('group', '?')}]{offline_mark}: {msg['text']}")
        elif t == "file_msg":
            data = base64.b64decode(msg["data_b64"])
            out_path = Path(f"./received_{msg['filename']}")
            out_path.write_bytes(data)
            print(f"[{msg['from']}] sent file: {msg['filename']} → saved as {out_path}")
        elif t == "user_joined":
            print(f"[server] {msg['username']} joined")
        elif t == "user_left":
            print(f"[server] {msg['username']} left")
        elif t == "users_list":
            print(f"[server] online: {', '.join(msg['users'])}")
        elif t == "groups_list":
            for g in msg["groups"]:
                print(f"  group {g['name']}: {len(g['members'])} members "
                      f"({', '.join(g['members'])})")
        elif t == "history":
            print(f"--- history with {msg['with_user']}: {len(msg['messages'])} messages ---")
            for m in msg["messages"]:
                print(f"  [{m['from']}] {m['text']}")
        elif t == "group_history":
            print(f"--- group {msg['group']} history: {len(msg['messages'])} messages ---")
            for m in msg["messages"]:
                print(f"  [{m['from']}] {m['text']}")
        elif t == "stats":
            print("--- server stats ---")
            for k, v in msg["data"].items():
                print(f"  {k}: {v}")
        elif t == "delivered":
            details = []
            if "online" in msg:
                details.append(f"online={msg['online']}")
            if "stored_offline" in msg and msg["stored_offline"]:
                details.append("stored offline")
            if "broadcast_to" in msg:
                details.append(f"broadcast to {msg['broadcast_to']}")
            if "online_recipients" in msg:
                details.append(f"group online={msg['online_recipients']}, offline={msg['offline_stored']}")
            if details:
                print(f"  ✓ {', '.join(details)}")
        elif t == "error":
            print(f"[error: {msg['code']}] {msg['message']}")
        elif t == "typing":
            print(f"... {msg['from']} is typing")
        elif t in ("group_created", "group_joined", "group_left", "pong"):
            print(f"[server] {t}: {msg.get('group', '')}")

    def run(self):
        recv_thread = threading.Thread(target=self._recv_loop, daemon=True)
        recv_thread.start()

        try:
            for line in sys.stdin:
                line = line.strip()
                if not line:
                    continue
                self._handle_input(line)
        except (EOFError, KeyboardInterrupt):
            pass
        self.alive = False
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            self.sock.close()
        except Exception:
            pass

    def _handle_input(self, line: str):
        if line == "/quit":
            self.alive = False
            return
        if line.startswith("/msg "):
            parts = line.split(" ", 2)
            if len(parts) < 3:
                print("usage: /msg <user> <text>")
                return
            self._send({"type": "private", "to": parts[1], "text": parts[2]})
        elif line.startswith("/all "):
            text = line[5:]
            self._send({"type": "broadcast", "text": text})
        elif line == "/users":
            self._send({"type": "list_users"})
        elif line.startswith("/group new "):
            name = line[len("/group new "):].strip()
            self._send({"type": "group_create", "group_name": name})
        elif line.startswith("/group join "):
            name = line[len("/group join "):].strip()
            self._send({"type": "group_join", "group_name": name})
        elif line.startswith("/group leave "):
            name = line[len("/group leave "):].strip()
            self._send({"type": "group_leave", "group_name": name})
        elif line.startswith("/group msg "):
            rest = line[len("/group msg "):]
            parts = rest.split(" ", 1)
            if len(parts) < 2:
                print("usage: /group msg <name> <text>")
                return
            self._send({"type": "group_msg", "group_name": parts[0], "text": parts[1]})
        elif line == "/groups":
            self._send({"type": "list_groups"})
        elif line.startswith("/history "):
            user = line[len("/history "):].strip()
            self._send({"type": "history", "with_user": user})
        elif line.startswith("/file "):
            parts = line.split(" ", 2)
            if len(parts) < 3:
                print("usage: /file <user> <path>")
                return
            user, path = parts[1], parts[2]
            try:
                data = Path(path).read_bytes()
                self._send({
                    "type": "file_send",
                    "to": user,
                    "filename": Path(path).name,
                    "data_b64": base64.b64encode(data).decode("ascii"),
                })
            except Exception as e:
                print(f"[error reading file] {e}")
        elif line.startswith("/typing "):
            user = line[len("/typing "):].strip()
            self._send({"type": "typing", "to": user})
        elif line == "/stats":
            self._send({"type": "stats"})
        else:
            print("Unknown command. Try /msg /all /users /group /history /file /stats /quit")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 cli_client.py <username> [host] [port]")
        sys.exit(1)
    username = sys.argv[1]
    host = sys.argv[2] if len(sys.argv) > 2 else "127.0.0.1"
    port = int(sys.argv[3]) if len(sys.argv) > 3 else 9000

    client = ChatClient(username, host, port)
    client.connect()
    print(f"Connected as {username} to {host}:{port}")
    print("Type /quit to exit, or use commands like /msg /all /users /group /history /file /stats")
    client.run()


if __name__ == "__main__":
    main()
