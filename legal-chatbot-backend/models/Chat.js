const mongoose = require('mongoose');

const chatSchema = new mongoose.Schema({
  username: { type: String, required: true, unique: true },
  chat_num: { type: Number, required: true}
}, {collection : 'chats'});

const Chat = mongoose.model('Chat', chatSchema);

module.exports = Chat;
