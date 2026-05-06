// Node.js TCP echo server — приймає JSON-повідомлення з Python-клієнта,
// логує число і відправляє його назад. Демонструє крос-мовний IPC.
const net = require('net');
const fs = require('fs');

const PORT = parseInt(process.env.PORT || '0', 10);
const LOG_PATH = process.env.LOG_PATH || '/tmp/node_helper.log';

// Очищуємо лог
fs.writeFileSync(LOG_PATH, '');

const server = net.createServer((socket) => {
    let buffer = '';
    let count = 0;

    socket.on('data', (chunk) => {
        buffer += chunk.toString('utf-8');
        // Кожне повідомлення закінчується \n
        let idx;
        while ((idx = buffer.indexOf('\n')) !== -1) {
            const line = buffer.slice(0, idx);
            buffer = buffer.slice(idx + 1);
            if (line === 'STOP') {
                fs.appendFileSync(LOG_PATH, `Total received: ${count}\n`);
                socket.write(JSON.stringify({ status: 'STOPPED', count: count }) + '\n');
                socket.end();
                // Закриваємо сервер і завершуємо явно
                server.close();
                setImmediate(() => process.exit(0));
                return;
            }
            try {
                const msg = JSON.parse(line);
                count++;
                // Логуємо у файл
                fs.appendFileSync(LOG_PATH, `[${count}] received: ${msg.value}\n`);
                // Ехо — повертаємо те саме значення
                socket.write(JSON.stringify({ id: msg.id, value: msg.value }) + '\n');
            } catch (e) {
                socket.write(JSON.stringify({ error: e.message }) + '\n');
            }
        }
    });
});

server.listen(PORT, '127.0.0.1', () => {
    const addr = server.address();
    // Виводимо порт у stdout — батьківський процес зчитує його
    console.log(`PORT=${addr.port}`);
});

// Грейсфул shutdown по сигналу
process.on('SIGTERM', () => { server.close(); process.exit(0); });
