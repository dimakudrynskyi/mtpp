"""
Чат-сервер: один процес, два мережеві інтерфейси:
  • TCP socket (порт 9000) — для CLI-клієнтів і автоматичних тестів.
  • WebSocket (порт 9001) — для веб-клієнтів у браузері.

Обидва інтерфейси спілкуються однаковим NDJSON-протоколом і працюють
на ОДНОМУ спільному ChatState (через тред-пара з RLock).

Архітектура
  Головний потік:
    1. Запускає WebSocket-сервер у власному asyncio thread.
    2. Запускає TCP-сервер у головному потоці: socket.accept() в циклі.
    3. Для кожного нового TCP-з'єднання — створює новий потік ClientHandler.
    4. Для кожного WS-з'єднання — створює ClientHandler у asyncio task.
"""
from __future__ import annotations
import asyncio
import json
import logging
import socket
import sys
import threading
import time
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from chat_state import ChatState, StoredMessage

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(threadName)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("chat_server")

# ============================================================================
# Базовий клієнт-handler (TCP). Кожен інстанс — це окремий потік.
# ============================================================================

class ClientHandler:
    """
    Обробник одного TCP-клієнта. Працює у власному потоці.
    """

    def __init__(self, conn: socket.socket, addr: tuple, state: ChatState):
        self.conn = conn
        self.addr = addr
        self.state = state
        self.username: Optional[str] = None
        self.send_lock = threading.Lock()   # серіалізує send() — TCP write
        self.alive = True

    def send_message(self, msg: dict):
        """Викликається з будь-якого потоку. Кладе в asyncio loop."""
        if not self.alive:
            raise ConnectionError("client is offline")
        line = json.dumps(msg, ensure_ascii=False)
        try:
            future = asyncio.run_coroutine_threadsafe(self.ws.send(line), self.loop)
            future.result(timeout=5.0)
        except Exception as e:
            self.alive = False
            # Закриваємо WebSocket щоб serve() вийшов з async for і виконав cleanup
            try:
                asyncio.run_coroutine_threadsafe(self.ws.close(), self.loop)
            except Exception:
                pass
            raise ConnectionError(f"WS send failed: {e}")

    def _send_error(self, code: str, message: str):
        try:
            self.send_message({"type": "error", "code": code, "message": message})
        except Exception:
            pass

    def serve(self):
        """Головний цикл обробки клієнта."""
        peer = f"{self.addr[0]}:{self.addr[1]}"
        log.info(f"+++ TCP client connected: {peer}")
        buffer = b""
        try:
            while self.alive:
                chunk = self.conn.recv(8192)
                if not chunk:
                    break
                buffer += chunk
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except json.JSONDecodeError as e:
                        self._send_error("BAD_JSON", str(e))
                        continue
                    self._handle_message(msg)
        except (ConnectionResetError, BrokenPipeError, OSError) as e:
            log.info(f"--- {peer} connection error: {e}")
        finally:
            self.alive = False
            if self.username:
                self.state.unregister_user(self.username)
                # Сповістити решту учасників про відключення
                self._broadcast_user_left()
            try:
                self.conn.close()
            except Exception:
                pass
            log.info(f"--- TCP client disconnected: {peer} ({self.username})")

    def _handle_message(self, msg: dict):
        """Маршрутизатор повідомлень за полем type."""
        msg_type = msg.get("type")
        # Перевіряємо реєстрацію — для більшості команд потрібен username
        needs_auth = msg_type not in ("register", "ping")
        if needs_auth and not self.username:
            self._send_error("NOT_REGISTERED", "Please register first")
            return

        try:
            handler_map = {
                "register": self._cmd_register,
                "ping": self._cmd_ping,
                "private": self._cmd_private,
                "broadcast": self._cmd_broadcast,
                "list_users": self._cmd_list_users,
                "group_create": self._cmd_group_create,
                "group_join": self._cmd_group_join,
                "group_leave": self._cmd_group_leave,
                "group_msg": self._cmd_group_msg,
                "list_groups": self._cmd_list_groups,
                "history": self._cmd_history,
                "group_history": self._cmd_group_history,
                "file_send": self._cmd_file_send,
                "typing": self._cmd_typing,
                "stats": self._cmd_stats,
            }
            fn = handler_map.get(msg_type)
            if fn is None:
                self._send_error("UNKNOWN_CMD", f"Unknown type: {msg_type}")
                return
            fn(msg)
        except (ConnectionError, BrokenPipeError, OSError) as e:
            # Клієнт відключений — нормальна ситуація, без трейсбеку
            log.info(f"client {self.username} disconnected during {msg_type}")
        except Exception as e:
            log.exception(f"Error handling {msg_type} from {self.username}")
            try:
                self._send_error("INTERNAL", str(e))
            except Exception:
                pass

    # ---------- commands ----------

    def _cmd_register(self, msg: dict):
        username = msg.get("username", "").strip()
        ok, message = self.state.register_user(username, self)
        if not ok:
            self._send_error("REGISTER_FAILED", message)
            return
        self.username = username
        self.send_message({
            "type": "welcome",
            "username": username,
            "server_time": time.time(),
        })
        # Сповістити інших про підключення
        with self.state.lock:
            other_handlers = [(n, h) for n, h in self.state.users.items()
                               if n != username]
        for _, h in other_handlers:
            try:
                h.send_message({"type": "user_joined", "username": username})
            except Exception:
                pass
        # Доставити офлайн-повідомлення
        offline = self.state.fetch_offline(username)
        for stored in offline:
            try:
                self.send_message({
                    "type": "user_msg" if stored.kind != "group" else "group_msg",
                    "from": stored.sender,
                    "to": stored.recipient,
                    "text": stored.text,
                    "timestamp": stored.timestamp,
                    "id": stored.msg_id,
                    "offline": True,
                })
            except Exception:
                break

    def _cmd_ping(self, msg: dict):
        self.send_message({"type": "pong"})

    def _cmd_private(self, msg: dict):
        recipient = msg.get("to")
        text = msg.get("text", "")
        delivered, msg_id, stored = self.state.deliver_private(
            self.username, recipient, text)
        self.send_message({
            "type": "delivered",
            "message_id": msg_id,
            "online": delivered,
            "stored_offline": stored,
        })

    def _cmd_broadcast(self, msg: dict):
        text = msg.get("text", "")
        count = self.state.broadcast(self.username, text)
        self.send_message({"type": "delivered", "broadcast_to": count})

    def _cmd_list_users(self, msg: dict):
        self.send_message({
            "type": "users_list",
            "users": self.state.get_online_users(),
        })

    def _cmd_group_create(self, msg: dict):
        group = msg.get("group_name")
        ok, message = self.state.create_group(self.username, group)
        if not ok:
            self._send_error("GROUP_CREATE_FAILED", message)
        else:
            self.send_message({"type": "group_created", "group": group})

    def _cmd_group_join(self, msg: dict):
        group = msg.get("group_name")
        ok, message = self.state.join_group(self.username, group)
        if not ok:
            self._send_error("GROUP_JOIN_FAILED", message)
        else:
            self.send_message({"type": "group_joined", "group": group})

    def _cmd_group_leave(self, msg: dict):
        group = msg.get("group_name")
        ok, message = self.state.leave_group(self.username, group)
        if not ok:
            self._send_error("GROUP_LEAVE_FAILED", message)
        else:
            self.send_message({"type": "group_left", "group": group})

    def _cmd_group_msg(self, msg: dict):
        group = msg.get("group_name")
        text = msg.get("text", "")
        online, offline = self.state.deliver_group(self.username, group, text)
        if online == -1:
            self._send_error("GROUP_NOT_FOUND", f"Group '{group}' does not exist")
            return
        self.send_message({
            "type": "delivered",
            "group": group,
            "online_recipients": online,
            "offline_stored": offline,
        })

    def _cmd_list_groups(self, msg: dict):
        self.send_message({
            "type": "groups_list",
            "groups": self.state.list_groups(),
        })

    def _cmd_history(self, msg: dict):
        with_user = msg.get("with_user")
        history = self.state.get_history(self.username, with_user)
        self.send_message({
            "type": "history",
            "with_user": with_user,
            "messages": history,
        })

    def _cmd_group_history(self, msg: dict):
        group = msg.get("group_name")
        history = self.state.get_group_history(group)
        self.send_message({
            "type": "group_history",
            "group": group,
            "messages": history,
        })

    def _cmd_file_send(self, msg: dict):
        recipient = msg.get("to")
        filename = msg.get("filename", "untitled")
        data_b64 = msg.get("data_b64", "")
        # Передаємо як спеціальне повідомлення; зберігаємо лише факт у історії
        marker = f"[FILE: {filename} ({len(data_b64)} b64 bytes)]"
        delivered, msg_id, stored = self.state.deliver_private(
            self.username, recipient, marker, kind="file")
        # Для онлайн-одержувачів — окреме файл-повідомлення з даними
        if delivered:
            handler = self.state.get_handler(recipient)
            if handler:
                try:
                    handler.send_message({
                        "type": "file_msg",
                        "from": self.username,
                        "filename": filename,
                        "data_b64": data_b64,
                        "timestamp": time.time(),
                        "id": msg_id,
                    })
                except Exception:
                    pass
        with self.state.lock:
            self.state.stats["files_sent"] += 1
        self.send_message({
            "type": "delivered",
            "message_id": msg_id,
            "online": delivered,
            "stored_offline": stored,
            "file": filename,
        })

    def _cmd_typing(self, msg: dict):
        recipient = msg.get("to")
        handler = self.state.get_handler(recipient)
        if handler:
            try:
                handler.send_message({"type": "typing", "from": self.username})
            except Exception:
                pass

    def _cmd_stats(self, msg: dict):
        self.send_message({"type": "stats", "data": self.state.get_stats()})

    def _broadcast_user_left(self):
        """При відключенні — сповістити інших."""
        with self.state.lock:
            handlers = list(self.state.users.values())
        for h in handlers:
            try:
                h.send_message({"type": "user_left", "username": self.username})
            except Exception:
                pass


# ============================================================================
# WebSocket-bridge: обгортка над WebSocket з тим самим інтерфейсом
# ============================================================================

class WebSocketHandler:
    """
    Обгортка над WebSocket-з'єднанням, що підтримує той самий API,
    що й TCP ClientHandler. send_message() може викликатись з будь-якого
    потоку — повідомлення кладеться в asyncio Queue, з якого asyncio task
    зчитує і надсилає.
    """

    def __init__(self, websocket, state: ChatState, loop: asyncio.AbstractEventLoop):
        self.ws = websocket
        self.state = state
        self.loop = loop
        self.username: Optional[str] = None
        self.alive = True
        # Внутрішня поведінка: переюзаємо TCP ClientHandler через міст.
        self._inner = _WSAdapter(self, state)

    def send_message(self, msg: dict):
        """Викликається з ЛЮБОГО потоку. Кладе в asyncio loop."""
        if not self.alive:
            raise ConnectionError("client is offline")
        line = json.dumps(msg, ensure_ascii=False)
        # call_soon_threadsafe — щоб надіслати з іншого потоку
        try:
            future = asyncio.run_coroutine_threadsafe(self.ws.send(line), self.loop)
            future.result(timeout=5.0)
        except Exception as e:
            self.alive = False
            raise

    async def serve(self):
        peer = self.ws.remote_address
        log.info(f"+++ WS client connected: {peer}")
        try:
            async for message in self.ws:
                try:
                    msg = json.loads(message)
                except json.JSONDecodeError as e:
                    await self.ws.send(json.dumps({
                        "type": "error", "code": "BAD_JSON", "message": str(e),
                    }))
                    continue
                # Виконуємо обробку синхронно через TCP ClientHandler логіку
                self._inner._handle_message(msg)
        except Exception as e:
            log.info(f"--- WS error: {e}")
        finally:
            self.alive = False
            if self.username:
                self.state.unregister_user(self.username)
                self._inner._broadcast_user_left()
            log.info(f"--- WS client disconnected: {peer} ({self.username})")


class _WSAdapter:
    """
    Адаптер, який вдає з себе ClientHandler для логіки команд,
    але передає send/унregister в зовнішній WebSocketHandler.
    """
    def __init__(self, ws_handler: WebSocketHandler, state: ChatState):
        self._wsh = ws_handler
        self.state = state

    @property
    def username(self):
        return self._wsh.username

    @username.setter
    def username(self, value):
        self._wsh.username = value

    def send_message(self, msg: dict):
        self._wsh.send_message(msg)

    def _send_error(self, code: str, message: str):
        try:
            self.send_message({"type": "error", "code": code, "message": message})
        except Exception:
            pass

    def _broadcast_user_left(self):
        with self.state.lock:
            handlers = list(self.state.users.values())
        for h in handlers:
            try:
                h.send_message({"type": "user_left", "username": self.username})
            except Exception:
                pass

    # Делегуємо всю обробку на ClientHandler-методи
    _handle_message = ClientHandler._handle_message
    _cmd_register = ClientHandler._cmd_register
    _cmd_ping = ClientHandler._cmd_ping
    _cmd_private = ClientHandler._cmd_private
    _cmd_broadcast = ClientHandler._cmd_broadcast
    _cmd_list_users = ClientHandler._cmd_list_users
    _cmd_group_create = ClientHandler._cmd_group_create
    _cmd_group_join = ClientHandler._cmd_group_join
    _cmd_group_leave = ClientHandler._cmd_group_leave
    _cmd_group_msg = ClientHandler._cmd_group_msg
    _cmd_list_groups = ClientHandler._cmd_list_groups
    _cmd_history = ClientHandler._cmd_history
    _cmd_group_history = ClientHandler._cmd_group_history
    _cmd_file_send = ClientHandler._cmd_file_send
    _cmd_typing = ClientHandler._cmd_typing
    _cmd_stats = ClientHandler._cmd_stats


# ============================================================================
# TCP server (потоки) + WebSocket server (asyncio в окремому потоці)
# ============================================================================

def run_tcp_server(host: str, port: int, state: ChatState, stop_event: threading.Event):
    """Стандартний TCP-сервер: для кожного клієнта — окремий потік."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(128)
    sock.settimeout(0.5)   # дозволяє перевіряти stop_event
    log.info(f"TCP server listening on {host}:{port}")

    try:
        while not stop_event.is_set():
            try:
                conn, addr = sock.accept()
            except socket.timeout:
                continue
            handler = ClientHandler(conn, addr, state)
            t = threading.Thread(target=handler.serve, daemon=True,
                                  name=f"tcp-{addr[1]}")
            t.start()
    finally:
        sock.close()
        log.info("TCP server stopped")


def run_websocket_server(host: str, port: int, state: ChatState,
                          stop_event: threading.Event):
    """
    WebSocket-сервер в окремому потоці. Усередині цього потоку — asyncio loop.
    """
    import websockets

    async def handler(websocket):
        loop = asyncio.get_event_loop()
        ws_handler = WebSocketHandler(websocket, state, loop)
        await ws_handler.serve()

    async def main():
        async with websockets.serve(handler, host, port,
                                      ping_interval=20, ping_timeout=10):
            log.info(f"WebSocket server listening on {host}:{port}")
            # Чекати на stop_event у фоні
            while not stop_event.is_set():
                await asyncio.sleep(0.5)

    asyncio.run(main())
    log.info("WebSocket server stopped")


def start_servers(tcp_port: int = 9000, ws_port: int = 9001,
                  host: str = "0.0.0.0") -> tuple[ChatState, threading.Event]:
    """Запускає TCP- і WS-сервери в окремих потоках. Повертає state і stop_event."""
    state = ChatState()
    stop_event = threading.Event()

    ws_thread = threading.Thread(
        target=run_websocket_server,
        args=(host, ws_port, state, stop_event),
        daemon=True, name="ws-server")
    ws_thread.start()

    tcp_thread = threading.Thread(
        target=run_tcp_server,
        args=(host, tcp_port, state, stop_event),
        daemon=True, name="tcp-server")
    tcp_thread.start()

    return state, stop_event


if __name__ == "__main__":
    import signal
    state, stop_event = start_servers()
    log.info("Server up. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        log.info("Shutting down...")
        stop_event.set()
        time.sleep(1)
