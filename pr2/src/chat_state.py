"""
Ядро чат-сервера: типи повідомлень, спільний стан, потокобезпечні операції.

Усі повідомлення між клієнтом і сервером — це JSON-об'єкти,
розділені символом '\\n' (NDJSON). Це простий і крос-мовний формат, який
легко парситься будь-яким клієнтом (Python, JavaScript, інші).

Типи повідомлень (поле "type"):
  Client → Server:
    register      {username}                   реєстрація
    private       {to, text}                   особисте повідомлення
    broadcast     {text}                       усім онлайн
    group_create  {group_name}                 створити групу
    group_join    {group_name}                 приєднатись
    group_leave   {group_name}                 покинути
    group_msg     {group_name, text}           повідомлення в групу
    file_send     {to, filename, data_b64}     передача файлу (base64)
    list_users    {}                           отримати список онлайн
    history       {with_user}                  історія розмови
    typing        {to}                         «друкує...» індикатор
    ping          {}                           перевірка з'єднання

  Server → Client:
    welcome       {username, server_time}       при успішній реєстрації
    error         {code, message}               помилка
    user_msg      {from, text, timestamp, id}   приватне повідомлення
    broadcast_msg {from, text, timestamp, id}   broadcast
    group_msg     {group, from, text, ts, id}   повідомлення в групі
    file_msg      {from, filename, data_b64}    отриманий файл
    user_joined   {username}                    хтось підключився
    user_left     {username}                    хтось відключився
    users_list    {users: [...]}                відповідь на list_users
    history       {with_user, messages: [...]}  історія
    typing        {from}                        індикатор друку
    pong          {}
    delivered     {message_id}                  підтвердження доставки

Сервер обслуговує кожного клієнта в окремому потоці.
Усі спільні структури (users, groups, message_log, offline_queues)
захищені одним threading.RLock через клас ChatState. RLock дозволяє
рекурсивне взяття замку всередині однієї нитки (потрібно при broadcast,
коли надсилання користувачу А може спричинити лог запис, що теж бере замок).
"""
from __future__ import annotations
import json
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Optional, Callable

# Максимальна кількість збережених повідомлень в історії на пару користувачів
MAX_HISTORY_PER_PAIR = 200
# Максимальна кількість офлайн-повідомлень на користувача
MAX_OFFLINE_QUEUE = 100


@dataclass
class StoredMessage:
    """Одне збережене повідомлення в журналі/історії/офлайн-черзі."""
    msg_id: int
    sender: str
    recipient: str            # ім'я користувача або префікс "group:NAME" або "*broadcast"
    text: str
    timestamp: float
    kind: str = "text"        # text | file | broadcast | group

    def to_dict(self) -> dict:
        return {
            "id": self.msg_id,
            "from": self.sender,
            "to": self.recipient,
            "text": self.text,
            "timestamp": self.timestamp,
            "kind": self.kind,
        }


class ChatState:
    """
    Спільний стан сервера. Усі мутації захищені self.lock.

    Атрибути:
      users        — dict[username → connection handler]
      groups       — dict[group_name → set[username]]
      offline_queues — dict[username → list[StoredMessage]]
      history      — dict[frozenset({user1, user2}) → deque[StoredMessage]]
      group_history — dict[group_name → deque[StoredMessage]]
      event_log    — deque останніх подій (для журналу/моніторингу)
      stats        — статистика: повідомлень доставлено, втрачено, тощо
    """

    def __init__(self):
        # ОДИН RLock на ВЕСЬ стан. Це coarse-grained locking — простий
        # і гарантовано без deadlock'ів. Для high-throughput можна було б
        # робити fine-grained по полях, але для навчальної задачі це зайве.
        self.lock = threading.RLock()
        self._next_msg_id = 1

        self.users: dict = {}               # username → ClientHandler
        self.groups: dict = defaultdict(set)
        self.offline_queues: dict = defaultdict(list)
        self.history: dict = {}             # frozenset({u1,u2}) → deque
        self.group_history: dict = defaultdict(lambda: deque(maxlen=MAX_HISTORY_PER_PAIR))
        self.event_log = deque(maxlen=1000)

        self.stats = {
            "messages_total": 0,
            "broadcasts": 0,
            "private_messages": 0,
            "group_messages": 0,
            "files_sent": 0,
            "offline_stored": 0,
            "offline_delivered": 0,
            "users_registered": 0,
            "disconnects": 0,
        }

    # ---------- low-level helpers ----------

    def _next_id(self) -> int:
        with self.lock:
            i = self._next_msg_id
            self._next_msg_id += 1
            return i

    def _log_event(self, kind: str, **fields):
        """Додає запис у журнал подій. Викликається під замком."""
        self.event_log.append({
            "kind": kind,
            "timestamp": time.time(),
            **fields,
        })

    def _pair_key(self, a: str, b: str) -> frozenset:
        return frozenset({a, b})

    # ---------- user lifecycle ----------

    def register_user(self, username: str, handler) -> tuple[bool, str]:
        """
        Реєструє користувача. Повертає (success, message).
        Якщо ім'я вже зайняте — повертає (False, "...").
        """
        with self.lock:
            if username in self.users:
                return False, f"Username '{username}' is already taken"
            if not username or "/" in username or len(username) > 32:
                return False, "Invalid username"
            self.users[username] = handler
            self.stats["users_registered"] += 1
            self._log_event("user_register", user=username)
            return True, "ok"

    def unregister_user(self, username: str):
        """Видаляє користувача (при відключенні). Безпечно навіть якщо немає."""
        with self.lock:
            if username in self.users:
                del self.users[username]
                self.stats["disconnects"] += 1
                self._log_event("user_disconnect", user=username)
                # Видалити з усіх груп
                for group_members in self.groups.values():
                    group_members.discard(username)

    def get_online_users(self) -> list[str]:
        with self.lock:
            return sorted(self.users.keys())

    def get_handler(self, username: str):
        with self.lock:
            return self.users.get(username)

    # ---------- private messaging ----------

    def deliver_private(self, sender: str, recipient: str, text: str,
                         kind: str = "text") -> tuple[bool, int, bool]:
        """
        Надіслати приватне повідомлення.
        Повертає (delivered_online, message_id, stored_offline).
        Якщо одержувач онлайн — викликає його handler.send(...).
        Інакше зберігає у offline_queue.
        """
        msg_id = self._next_id()
        ts = time.time()
        stored = StoredMessage(msg_id=msg_id, sender=sender, recipient=recipient,
                                text=text, timestamp=ts, kind=kind)

        with self.lock:
            # Зберегти в історії пари користувачів
            key = self._pair_key(sender, recipient)
            if key not in self.history:
                self.history[key] = deque(maxlen=MAX_HISTORY_PER_PAIR)
            self.history[key].append(stored)

            self.stats["messages_total"] += 1
            self.stats["private_messages"] += 1
            self._log_event("private_msg", sender=sender, recipient=recipient,
                             id=msg_id)

            handler = self.users.get(recipient)
            if handler is None:
                # Офлайн — зберігаємо у чергу
                queue = self.offline_queues[recipient]
                if len(queue) < MAX_OFFLINE_QUEUE:
                    queue.append(stored)
                    self.stats["offline_stored"] += 1
                return False, msg_id, True

        # Спочатку відпускаємо замок, потім викликаємо handler
        # (handler може блокуватись на сокеті — не хочемо тримати замок)
        try:
            handler.send_message({
                "type": "user_msg" if kind == "text" else "file_msg",
                "from": sender,
                "text": text,
                "timestamp": ts,
                "id": msg_id,
            })
            return True, msg_id, False
        except Exception:
            # Помилка надсилання — fallback в offline-чергу
            with self.lock:
                queue = self.offline_queues[recipient]
                if len(queue) < MAX_OFFLINE_QUEUE:
                    queue.append(stored)
                    self.stats["offline_stored"] += 1
            return False, msg_id, True

    def fetch_offline(self, username: str) -> list[StoredMessage]:
        """Забрати всі офлайн-повідомлення для користувача."""
        with self.lock:
            messages = self.offline_queues.pop(username, [])
            if messages:
                self.stats["offline_delivered"] += len(messages)
                self._log_event("offline_delivered", user=username,
                                 count=len(messages))
            return messages

    # ---------- broadcast ----------

    def broadcast(self, sender: str, text: str) -> int:
        """Надіслати всім онлайн (крім самого відправника). Повертає кількість одержувачів."""
        msg_id = self._next_id()
        ts = time.time()
        with self.lock:
            recipients = [(name, h) for name, h in self.users.items() if name != sender]
            self.stats["messages_total"] += 1
            self.stats["broadcasts"] += 1
            self._log_event("broadcast", sender=sender, count=len(recipients),
                             id=msg_id)

        # Надсилаємо ПОЗА замком — щоб повільні клієнти не блокували систему
        delivered = 0
        for _, handler in recipients:
            try:
                handler.send_message({
                    "type": "broadcast_msg",
                    "from": sender,
                    "text": text,
                    "timestamp": ts,
                    "id": msg_id,
                })
                delivered += 1
            except Exception:
                pass
        return delivered

    # ---------- groups ----------

    def create_group(self, owner: str, group_name: str) -> tuple[bool, str]:
        with self.lock:
            if group_name in self.groups:
                return False, f"Group '{group_name}' already exists"
            if not group_name or len(group_name) > 32 or "/" in group_name:
                return False, "Invalid group name"
            self.groups[group_name] = {owner}
            self._log_event("group_create", owner=owner, group=group_name)
            return True, "ok"

    def join_group(self, username: str, group_name: str) -> tuple[bool, str]:
        with self.lock:
            if group_name not in self.groups:
                return False, f"Group '{group_name}' does not exist"
            self.groups[group_name].add(username)
            self._log_event("group_join", user=username, group=group_name)
            return True, "ok"

    def leave_group(self, username: str, group_name: str) -> tuple[bool, str]:
        with self.lock:
            if group_name in self.groups:
                self.groups[group_name].discard(username)
                self._log_event("group_leave", user=username, group=group_name)
                return True, "ok"
            return False, f"Group '{group_name}' does not exist"

    def deliver_group(self, sender: str, group_name: str, text: str) -> tuple[int, int]:
        """Надіслати в групу. Повертає (online_recipients, offline_stored)."""
        msg_id = self._next_id()
        ts = time.time()

        with self.lock:
            if group_name not in self.groups:
                return -1, 0
            members = list(self.groups[group_name])
            online_handlers = []
            offline_users = []
            for member in members:
                if member == sender:
                    continue
                h = self.users.get(member)
                if h is not None:
                    online_handlers.append((member, h))
                else:
                    offline_users.append(member)

            stored = StoredMessage(msg_id=msg_id, sender=sender,
                                    recipient=f"group:{group_name}",
                                    text=text, timestamp=ts, kind="group")
            self.group_history[group_name].append(stored)

            # Зберегти для офлайн-членів
            for member in offline_users:
                queue = self.offline_queues[member]
                if len(queue) < MAX_OFFLINE_QUEUE:
                    queue.append(stored)
                    self.stats["offline_stored"] += 1

            self.stats["messages_total"] += 1
            self.stats["group_messages"] += 1
            self._log_event("group_msg", sender=sender, group=group_name,
                             online=len(online_handlers),
                             offline=len(offline_users), id=msg_id)

        # Поза замком
        delivered = 0
        for _, handler in online_handlers:
            try:
                handler.send_message({
                    "type": "group_msg",
                    "group": group_name,
                    "from": sender,
                    "text": text,
                    "timestamp": ts,
                    "id": msg_id,
                })
                delivered += 1
            except Exception:
                pass
        return delivered, len(offline_users)

    def list_groups(self) -> list[dict]:
        with self.lock:
            return [{"name": name, "members": sorted(list(members))}
                    for name, members in self.groups.items()]

    # ---------- history ----------

    def get_history(self, user_a: str, user_b: str) -> list[dict]:
        with self.lock:
            key = self._pair_key(user_a, user_b)
            history = self.history.get(key, [])
            return [m.to_dict() for m in history]

    def get_group_history(self, group_name: str) -> list[dict]:
        with self.lock:
            return [m.to_dict() for m in self.group_history.get(group_name, [])]

    # ---------- stats / monitoring ----------

    def get_stats(self) -> dict:
        with self.lock:
            return {
                **self.stats,
                "users_online": len(self.users),
                "groups": len(self.groups),
                "queued_offline": sum(len(q) for q in self.offline_queues.values()),
                "history_pairs": len(self.history),
            }

    def get_recent_events(self, n: int = 100) -> list[dict]:
        with self.lock:
            return list(self.event_log)[-n:]
