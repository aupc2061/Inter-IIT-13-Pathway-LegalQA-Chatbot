const mongoose = require('mongoose');

// Define the user schema
const userSchema = new mongoose.Schema({
  username: {
    type: String,
    required: true,
    unique: true,  // Ensure usernames are unique
  },
  password: {
    type: String,
    required: true,
  },
  name: {  // Add the name field
    type: String,
    required: true,  // Optional, set to true if you want the name to be required
  }
}, { collection: "users", timestamps: true });  // Automatically add createdAt and updatedAt fields

// Create the model
const User = mongoose.model('User', userSchema);

module.exports = User;  // Export the model
