const { contextBridge, ipcRenderer } = require('electron');

contextBridge.exposeInMainWorld('api', {
  getCustomers: () => ipcRenderer.invoke('get-customers'),
  addCustomer: (customer) => ipcRenderer.invoke('add-customer', customer),
  getCustomerTransactions: (customerId) => ipcRenderer.invoke('get-customer-transactions', customerId),
  addCustomerTransaction: (txn) => ipcRenderer.invoke('add-customer-transaction', txn),
  getShopTransactions: () => ipcRenderer.invoke('get-shop-transactions'),
  addShopTransaction: (txn) => ipcRenderer.invoke('add-shop-transaction', txn),
  getMeta: (key) => ipcRenderer.invoke('get-meta', key),
  setMeta: (key, value) => ipcRenderer.invoke('set-meta', key, value)
});
