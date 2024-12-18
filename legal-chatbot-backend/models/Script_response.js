const mongoose = require('mongoose');

// Define the schema
const ScriptResponse = new mongoose.Schema({
  username: {
    type: String,
    required: true,
  },
  chat_num: {
    type: Number, // Changed from String 
    required: true,
  },
  scriptResponse: {
    type: Object, // Flexible to store any JSON object
    required: true,
  },
  usermessage: {
    type: String,
    required: true,
  },
  createdAt: {
    type: Date,
    default: Date.now, // Automatically adds the timestamp when a document is created
  },
}, {collection: "script_response"});

// Create the model
const Script_response = mongoose.model('ScriptResponse', ScriptResponse);

module.exports = Script_response;