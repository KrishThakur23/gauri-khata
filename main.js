const { app, BrowserWindow, ipcMain } = require('electron');
const path = require('path');
const fs = require('fs');
const initSqlJs = require('sql.js');

let db;
let SQL;
let dbFilePath;

function createWindow () {
  const win = new BrowserWindow({
    width: 800,
    height: 600,
    webPreferences: {
      preload: path.join(__dirname, 'preload.js'),
      nodeIntegration: false,
      contextIsolation: true
    }
  });

  win.loadFile('index.html');
}

app.whenReady().then(async () => {
  SQL = await initSqlJs();
  dbFilePath = path.join(app.getPath('userData'), 'app.sqlite');
  let dbData = null;
  if (fs.existsSync(dbFilePath)) {
    dbData = new Uint8Array(fs.readFileSync(dbFilePath));
  }
  db = new SQL.Database(dbData);

  // --- Create tables ---
  db.run(`CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    phone TEXT,
    pan TEXT,
    notes TEXT,
    cashBalance REAL,
    goldBalance REAL,
    silverBalance REAL,
    photo_path TEXT,
    aadhar_front_path TEXT,
    aadhar_back_path TEXT
  )`);
  db.run(`CREATE TABLE IF NOT EXISTS customer_transactions (
    id TEXT PRIMARY KEY,
    customer_id INTEGER,
    timestamp TEXT,
    category TEXT,
    details TEXT,
    cashBalanceAfter REAL,
    cashChange REAL,
    goldBalanceAfter REAL,
    goldChange REAL,
    silverBalanceAfter REAL,
    silverChange REAL,
    FOREIGN KEY(customer_id) REFERENCES customers(id)
  )`);
  db.run(`CREATE TABLE IF NOT EXISTS shop_transactions (
    id TEXT PRIMARY KEY,
    timestamp TEXT,
    category TEXT,
    details TEXT
  )`);
  db.run(`CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT
  )`);

  function saveDb() {
    const data = db.export();
    fs.writeFileSync(dbFilePath, Buffer.from(data));
  }

  // --- IPC Handlers ---
  ipcMain.handle('get-customers', () => {
    const stmt = db.prepare('SELECT * FROM customers');
    const customers = [];
    while (stmt.step()) {
      customers.push(stmt.getAsObject());
    }
    stmt.free();
    return customers;
  });

  ipcMain.handle('add-customer', (event, customer) => {
    const insert = db.prepare(`INSERT INTO customers (name, phone, pan, notes, cashBalance, goldBalance, silverBalance, photo_path, aadhar_front_path, aadhar_back_path) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
    insert.run([
      customer.name,
      customer.phone,
      customer.pan,
      customer.notes || '',
      customer.cashBalance || 0,
      customer.goldBalance || 0,
      customer.silverBalance || 0,
      customer.photo_path || '',
      customer.aadhar_front_path || '',
      customer.aadhar_back_path || ''
    ]);
    saveDb();
    const id = db.prepare('SELECT last_insert_rowid() as id');
    id.step();
    const newId = id.getAsObject().id;
    id.free();
    return { ...customer, id: newId };
  });

  ipcMain.handle('get-customer-transactions', (event, customer_id) => {
    const stmt = db.prepare('SELECT * FROM customer_transactions WHERE customer_id = ?');
    const txns = [];
    stmt.bind([customer_id]);
    while (stmt.step()) {
      txns.push(stmt.getAsObject());
    }
    stmt.free();
    return txns;
  });

  ipcMain.handle('add-customer-transaction', (event, txn) => {
    const insert = db.prepare(`INSERT INTO customer_transactions (id, customer_id, timestamp, category, details, cashBalanceAfter, cashChange, goldBalanceAfter, goldChange, silverBalanceAfter, silverChange) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`);
    insert.run([
      txn.id,
      txn.customer_id,
      txn.timestamp,
      txn.category,
      JSON.stringify(txn.details),
      txn.cashBalanceAfter,
      txn.cashChange,
      txn.goldBalanceAfter,
      txn.goldChange,
      txn.silverBalanceAfter,
      txn.silverChange
    ]);
    saveDb();
    return txn;
  });

  ipcMain.handle('get-shop-transactions', () => {
    const stmt = db.prepare('SELECT * FROM shop_transactions');
    const txns = [];
    while (stmt.step()) {
      let row = stmt.getAsObject();
      row.details = row.details ? JSON.parse(row.details) : {};
      txns.push(row);
    }
    stmt.free();
    return txns;
  });

  ipcMain.handle('add-shop-transaction', (event, txn) => {
    const insert = db.prepare('INSERT INTO shop_transactions (id, timestamp, category, details) VALUES (?, ?, ?, ?)');
    insert.run([
      txn.id,
      txn.timestamp,
      txn.category,
      JSON.stringify(txn.details)
    ]);
    saveDb();
    return txn;
  });

  ipcMain.handle('get-meta', (event, key) => {
    const stmt = db.prepare('SELECT value FROM meta WHERE key = ?');
    stmt.bind([key]);
    if (stmt.step()) {
      const value = stmt.getAsObject().value;
      stmt.free();
      return value;
    }
    stmt.free();
    return null;
  });

  ipcMain.handle('set-meta', (event, key, value) => {
    const upsert = db.prepare('INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)');
    upsert.run([key, value]);
    saveDb();
    return true;
  });

  createWindow();

  app.on('activate', function () {
    if (BrowserWindow.getAllWindows().length === 0) createWindow();
  });
});

app.on('window-all-closed', function () {
  if (process.platform !== 'darwin') app.quit();
});
