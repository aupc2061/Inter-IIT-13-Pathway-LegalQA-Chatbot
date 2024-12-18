const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const { execFile } = require('child_process');
const mongoose = require('mongoose');
const User = require('./models/User'); // MongoDB Model for Users
const Chat = require('./models/Chat'); // MongoDB Model for storing chat history
const Script_response = require('./models/Script_response')
const session = require('express-session'); // For session management
const path = require('path');
const cookieParser = require('cookie-parser'); // Cookie parser for handling cookies
const { Script } = require('vm');

const app = express();
const port = process.env.PORT || 5000;

// Middleware setup
app.use(
  cors({
    origin: 'http://localhost:5173', // Replace with the frontend's URL
    credentials: true,  // Allow credentials (cookies) to be sent and received
  })
);
app.use(bodyParser.json()); // Parse incoming JSON requests
app.use(cookieParser()); // Parse cookies

// Session management middleware
app.use(
  session({
    secret: 'your-secret-key', // Replace with a stronger secret in production
    resave: false,
    saveUninitialized: true,
    cookie: {
      secure: false,  // Set to true in production if using https
    },
  })
);

const WebSocket = require('ws');
const fs = require('fs');

// Create WebSocket server
const wss = new WebSocket.Server({ port: 5001 });

// Handle WebSocket connection
wss.on('connection', (ws) => {
  console.log('New WebSocket connection');
  const cacheFile = 'cache1.txt';
  // Monitor cache.txt for changes
  fs.watch(cacheFile, (eventType) => {
    if (eventType === 'change') {
      fs.readFile(cacheFile, 'utf8', (err, data) => {
        if (err) {
          console.error('Error reading cache.txt:', err);
          return;
        }
        
        // If the feedback flag is set to 'True', notify frontend
        if (data.includes('flag = True')) {
          ws.send('triggerPopup'); // Notify frontend to show the feedback popup
        }
      });
    }
  });
});

console.log('WebSocket server listening on ws://localhost:5001');



//mongo db connection
mongoose.connect('insert mongo URL here', {
  useNewUrlParser: true,
  useUnifiedTopology: true,
})
  .then(() => console.log('Connected to MongoDB'))
  .catch((error) => console.error('MongoDB connection error:', error));


  app.post('/api/submit-feedback', async (req, res) => {
    console.log('Received feedback:', req.body);
    const { feedback } = req.body;
  
    // Validate the feedback to ensure it's not empty
    if (!feedback || feedback.trim() === '') {
      return res.status(400).json({ message: 'Feedback cannot be empty' });
    }
    const cacheFile = 'cache1.txt';
    console.log("cache file path in server.js:", cacheFile);
  
    try {
      // Create the content that will be written into cache.txt
      const fileContent = `flag = False\nfeedback = '${feedback}'`;
  
      // Write the new content to cache.txt, replacing any previous content
      fs.writeFile(cacheFile, fileContent, (err) => {
        console.log('Feedback written to cache.txt');
        if (err) {
          console.error('Error writing to cache.txt:', err);
          return res.status(500).json({ message: 'Failed to update cache.txt' });
        }
      
        // Send success response back to client
        res.json({ message: 'Feedback submitted successfully' });
      });

      
      fs.writeFile('text_files/human_input_node.txt', feedback, (err) => {
        if (err) {
          console.error('Error writing to file:', err);
        } else {
          console.log('File written successfully');
        }
      });




    } catch (error) {
      console.error('Error submitting feedback:', error);
      res.status(500).json({ message: 'Error submitting feedback' });
    }
  });
  
  
  
// Create new chat and increment chat ID
app.post('/api/new-chat', async (req, res) => {
  try {
    const username = req.session.user.username;  // Get username from session

    if (!username) {
      return res.status(401).json({ message: 'User not authenticated' });
    }

    let new_chat; // Initialize the variable to hold the chat object

    // Find the latest chat for the user and increment the chat_num
    const chat = await Chat.findOne({ username: username }); // Use `await` to wait for the query result

    if (chat) {
      // If a chat is found, increment the chat_num
      const new_chat_num = chat.chat_num + 1;
      chat.chat_num = new_chat_num; // Update chat_num
      new_chat = chat; // Set the updated chat
     // console.log("Updated chat number for user:", new_chat.chat_num);  // Accessing the updated `chat_num`
    } else {
      // If no chat is found, create a new chat with chat_num = 1
      new_chat = new Chat({
        username,
        chat_num: 1
      });
      console.log("No previous chat found for this user, creating new chat.");
    }

    // Save the updated or new chat document
    await new_chat.save();  

    res.json(new_chat);  // Send the new or updated chat document as a JSON response
    console.log('Chat created or updated with chat_num:', new_chat.chat_num);
  } catch (error) {
    console.error('Error creating new chat:', error);
    res.status(500).json({ message: 'Error creating new chat' });
  }
});



// Chatbot interaction endpoint
app.post('/api/chat', async (req, res) => {
  const userMessage = req.body.message;
  const username = req.session.user.username; // Assuming the username is sent with the message
  const chatID = parseInt(req.body.chatID);
 // console.log("chatid in chat api:", chatID);

  console.log('Received user message:', userMessage);

  // Define the path to the Python script
  const scriptPath = '..\\legal-chatbot-frontend\\src\\final_scripts\\main.py';
 //const scriptPath = 'legal-chatbot-frontend/src/final_scripts/main.py';
 // console.log('Script path:', scriptPath);
  // Safely pass arguments to the script
  execFile('python', [scriptPath, userMessage, username, chatID], async (error, stdout) => {
    console.log('Python script executed');
    if (error) {
      console.error('Error executing script:', error);
      return res.status(500).json({ response: 'Error processing query' });
    }

    const botResponse = stdout.trim(); // Assuming the Python script uses `print` for its output

    res.json({ response: botResponse });
    console.log('Bot response in chat api:', botResponse);
  });
});

// Get chat history for the logged-in user
app.get('/api/getChats', async (req, res) => {
  console.log("entered getChats");
  const chatID = req.query.chatID;
  const username = req.session.user.username; // Get the username from the session
 // console.log("username in getChats: ", username);
 // console.log("chatID in getChats: ", chatID);
  if (!username) {
    return res.status(401).json({ message: 'User not authenticated' });
  }
  if (!chatID) {
    return res.status(400).json({ message: 'Chat ID is required' });
  }

  try {
    // Convert chatID to a number (if necessary)
    const chat_num = parseInt(chatID, 10);

    if (isNaN(chat_num)) {
      return res.status(400).json({ message: 'Invalid Chat ID' });
    }

    //console.log("chat_num in getChats: ", chat_num);

    // Fetch chat history for the logged-in user and chat_num
    const script_res = await Script_response.find({ username: username, chat_num: chat_num });

    if (script_res.length === 0) {
      return res.status(404).json({ message: 'No chat history found for this user' });
    }

   // console.log('Chat history retrieved for user:', username);
    res.json({ chats: script_res }); // Send the chat data as a JSON response
    //console.log("script response: ", script_res);
  } catch (err) {
    console.error('Error retrieving chat history:', err);
    res.status(500).json({ message: 'Error retrieving chat history' });
  }
});

// GET API to fetch user chats
app.get('/api/getUserChats', async (req, res) => {
  console.log("entered getuserchats");
  const username = req.session.user.username; // Get the username from the session
  if (!username) {
    return res.status(401).json({ error: 'User not logged in' });
  }
  console.log("username in getuserchats: ", username);

  try {
    // Find all chats related to the username and sort by time (assuming `timestamp` is available in scriptResponse)
    const chats = await Script_response.find({ username: username }) // Query chats by username
      .sort({ 'scriptResponse.timestamp': 1 }) // Sort by timestamp, ascending order (adjust if needed)
      .select('scriptResponse usermessage chat_num username') // Only select the required fields
      .exec();

    if (!chats || chats.length === 0) {
      return res.status(404).json({ message: 'No chats found for this user' });
    }

    // Process chats to return user messages and final answers
    const processedChats = chats.map(chat => {
      const { scriptResponse, usermessage, chat_num, username } = chat;

      // Extracting final_ans from scriptResponse (scriptResponse is an object, not an array)
      let final_ans = null;
      //console.log("scriptResponse in getuserchats:", scriptResponse);

      if (scriptResponse && scriptResponse.final_ans) {
        final_ans = scriptResponse.final_ans; // Directly accessing final_ans from the object
      }

      return {
        chat_num,  // Return the chat number
        username,  // Return the username
        usermessage,  // Return the user's message
        final_ans,  // Return the final answer
      };
    });
    //console.log('Processed chats in getuserchats:', processedChats);
    res.json({ chats: processedChats });
  } catch (error) {
    console.error('Error fetching chats:', error);
    res.status(500).json({ error: 'Internal server error' });
  }
});


// Sign Up Route (MongoDB)
app.post('/api/signup', async (req, res) => {
  const { username, password, name } = req.body;

  // Check if user already exists
  const userExists = await User.findOne({ username });
  if (userExists) {
    return res.status(400).json({ message: 'User already exists.' });
  }

  // Store the new user in MongoDB
  const user = new User({ username, password, name });
  await user.save();
  res.status(201).json({ message: 'User registered successfully.' });
});

// Sign In Route (MongoDB)
app.post('/api/signin', async (req, res) => {
  const { username, password } = req.body;

  const user = await User.findOne({ username });
  if (!user || user.password !== password) {
    return res.status(401).json({ message: 'Invalid credentials' });
  }

  // Store user info in session
  req.session.user = { username: username, name: user.name };
  res.json({message: 'Sign in successful', user: req.session.user});
});

// Sign Out Route (to destroy the session and log the user out)
app.post('/api/signout', (req, res) => {
  // Destroy the session, effectively logging the user out
  req.session.destroy((err) => {
    if (err) {
      console.error('Error during session destruction:', err);
      return res.status(500).json({ message: 'Error during sign-out' });
    }

    // After session is destroyed, respond with a success message
    res.clearCookie('connect.sid'); // Optionally, clear the session cookie if using cookie-based session management
    res.json({ message: 'Sign out successful' });

    // Optionally, redirect the user to the home page or a login page (in case you want to redirect to a different page)
    // res.redirect('/');
  });
});


// Route to get currently signed-in user
app.get('/api/current-user', (req, res) => {
  console.log('Current session:', req.session);
  console.log('Current user in  current user called:', req.session.user);
  if (req.session.user) {
    res.json({ user: req.session.user });
  } else {
    res.status(401).json({ message: 'No user is signed in' });
  }
});

app.get('/api/get-thought-process', (req, res) => {
  const dirname = process.cwd();
  const folderPath = path.join(dirname, 'text_files');
  const files = fs.readdirSync(folderPath); // Read the folder contents

  //console.log('Files:', files);
  

  let thoughtProcess = [];

  files.forEach(file => {
    const filePath = path.join(folderPath, file);
    const content = fs.readFileSync(filePath, 'utf-8').trim(); // Read file content
   // console.log('Content:', content);
    if (content) {
      thoughtProcess.push({ type: file.replace('.txt', ''), text: content });
    }
  });
  //console.log('Thought process:', thoughtProcess);
  res.json({ thoughtProcess });
});

app.post('/api/empty-text-files', (req, res) => {
  const dirname = process.cwd();
  const folderPath = path.join(dirname, 'text_files');
  const files = fs.readdirSync(folderPath); // Read the folder contents

  files.forEach(file => {
    const filePath = path.join(folderPath, file);
    fs.writeFileSync(filePath, ''); // Empty the file
  });

  res.json({ message: 'Text files emptied successfully' });
}
);





// Start the server
app.listen(port, () => {
  console.log(`Server running on port ${port}`);
});
